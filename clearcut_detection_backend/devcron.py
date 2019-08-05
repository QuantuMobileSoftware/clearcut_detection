#!/usr/bin/env python

# Uses Brian's python cron code from
# http://stackoverflow.com/questions/373335/suggestions-for-a-cron-like-scheduler-in-python


from datetime import datetime
import math
import logging
import optparse
from subprocess import Popen
import sys
import time
import six
import re


def main():
    prog = "devcron.py"
    usage = "usage: %prog [options] crontab"
    description = "A development cron daemon. See README.md for more info."

    op = optparse.OptionParser(prog=prog, usage=usage, description=description)
    op.add_option("-v", "--verbose", dest="verbose", action="store_true",
                  help="verbose logging.")

    (options, args) = op.parse_args()

    if len(args) != 1:
        op.print_help()
        sys.exit(1)

    log_level = logging.WARN
    if options.verbose:
        log_level = logging.DEBUG

    logging.basicConfig(level=log_level)

    crontab_data = open(args[0]).read()
    crontab_data = fold_crontab_lines(crontab_data)
    crontab_data = edit_crontab_data(crontab_data)
    logging.debug("Edited crontab looks like:\n%s\n" % crontab_data)
    events = parse_crontab(crontab_data)
    logging.debug("Parsed crontab as:\n%s\n" %
                  "\n".join([str(e) for e in events]))
    cron = Cron(events)
    cron.run()


def fold_crontab_lines(data):
    return data.replace("\\\n", "")


def edit_crontab_data(data):
    deletions = []
    for line in data.splitlines():
        delete_cmd = "# devcron delete_str "
        if line.startswith(delete_cmd):
            if line[-1] == " ":
                logging.warn("There is a significant trailing space on line "
                             "'%s'." % line)
            deletions.append(line[len(delete_cmd):])
    logging.debug("Deleting the following strings: %s\n" % deletions)
    for d in deletions:
        data = data.replace(d, "")
    return data


def parse_crontab(data):
    """
    Returns a list of Events, one per line.
    """
    events = []
    for line in data.splitlines():
        line = line.strip()
        logging.debug("Parsing crontab line: %s" % line)
        if len(line) == 0 or line[0] == "#":
            continue
        if line[0] == "@":
            freq, cmd = line[1:].split(None, 1)
            if freq == "weekly":
                event = Event(make_cmd_runner(cmd), 0, 0, dow=0)
            else:
                raise NotImplementedError()
                # @ToDo:
        else:
            chunks = line.split(None, 5)
            if len(chunks) < 6:
                raise NotImplementedError("The crontab line must have 6 fields")
            event = Event(make_cmd_runner(chunks[5]),
                          parse_arg(chunks[0]),
                          parse_arg(chunks[1]),
                          parse_arg(chunks[2]),
                          parse_arg(chunks[3]),
                          parse_arg(chunks[4],
                                    lambda dow: 7 if dow == 0 else dow))
        events.append(event)
    return events


def make_cmd_runner(cmd):
    """
    Takes a path to a cmd and returns a function that when called,
    will run it.
    """
    def run_cmd():
        return Popen(cmd, shell=True, close_fds=True)
    run_cmd.__doc__ = cmd
    return run_cmd


def no_change(x):
    return x


_arg_regex = re.compile(r"""(?:
                              ((?:(?:\d+(?:-\d+)?),?)+)  # numbers and ranges divied by commas (1)
                              |                          # or
                              (\*)                       # asterisk (2)
                            )
                            (?:/(\d+))?                 # divisor part (3 - number part)
                            $""",
                        re.VERBOSE)


def parse_number_or_range(number_or_range):
    if "-" in number_or_range:
        start, end = (int(x) for x in number_or_range.split("-", 1))
        return range(start, end + 1)
    else:
        return [int(number_or_range)]


def parse_set_of_ranges(set_of_ranges):
    values = []
    for number_or_range in set_of_ranges.split(","):
        values.extend(parse_number_or_range(number_or_range))
    return values


def parse_arg(arg, converter=no_change):
    """
    This takes a crontab time arg and converts it to a python int, iterable,
    or set.
    If a callable is passed as converter, it translates numbers in arg with
    the converter.
    """
    match = _arg_regex.match(arg)
    if not match:
        raise NotImplementedError("The crontab line is malformed or isn't "
                                  "supported.")

    divisor = int(match.group(3)) if match.group(3) else 1

    if match.group(2):  # asterisk
        return DivisableMatch(divisor)

    values = parse_set_of_ranges(match.group(1))
    return [converter(int(n))
            for i, n in enumerate(values)
            if i % divisor == 0]


class DivisableMatch(set):
    """
    Matches X if (X-offset) % divisor == 0
    """
    def __init__(self, divisor, offset=0):
        self.divisor = divisor
        self.offset = offset

    def __contains__(self, item):
        return (int(item) - self.offset) % self.divisor == 0


all_match = DivisableMatch(1)


def convert_to_set(obj):
    if isinstance(obj, six.integer_types):
        return set([obj])
    if not isinstance(obj, set):
        obj = set(obj)
    return obj


class Event(object):
    def __init__(self, action, min=all_match, hour=all_match,
                 day=all_match, month=all_match, dow=all_match,
                 args=(), kwargs={}):
        """
        day: 1 - num days
        month: 1 - 12
        dow: mon = 1, sun = 7
        """
        self.mins = convert_to_set(min)
        self.hours = convert_to_set(hour)
        self.days = convert_to_set(day)
        self.months = convert_to_set(month)
        self.dow = convert_to_set(dow)
        self.action = action
        self.args = args
        self.kwargs = kwargs

    def matchtime(self, t):
        """
        Return True if this event should trigger at the specified datetime
        """
        return ((t.minute in self.mins) and
                (t.hour in self.hours) and
                (t.day in self.days) and
                (t.month in self.months) and
                (t.isoweekday() in self.dow))

    def check(self, t):
        if self.matchtime(t):
            self.process = self.action(*self.args, **self.kwargs)

    def __str__(self):
        return ("Event(%s, %s, %s, %s, %s, %s)" %
                (self.mins, self.hours, self.days, self.months, self.dow,
                 self.action.__doc__))


class Cron(object):
    step = 60

    def __init__(self, events):
        self.events = events

    def run(self, stop_condition=lambda: False):
        next_event = math.floor(time.time() / self.step) * self.step
        while not stop_condition():
            for e in self.events:
                e.check(datetime.fromtimestamp(next_event))

            next_event += self.step
            now = time.time()
            while now < next_event:
                dt = next_event - now
                logging.debug("Sleeping from %s to %s (%s secs)" % (
                    datetime.fromtimestamp(now),
                    datetime.fromtimestamp(next_event),
                    dt
                ))
                time.sleep(dt)
                now = time.time()


if __name__ == "__main__":
    main()
