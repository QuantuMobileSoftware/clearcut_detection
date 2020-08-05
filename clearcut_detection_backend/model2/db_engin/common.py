import sys
import traceback
import pytz
from datetime import datetime
from decimal import Decimal
from sqlalchemy.exc import IntegrityError, InternalError
from sqlalchemy.dialects import postgresql


class DataBaseResponseMixin:

    @staticmethod
    def get_single_obj(session, text_obj):
        """
        Handles requests that return one field
        :return:
        """
        try:
            print('text_obj.compile =', text_obj.compile(dialect=postgresql.dialect()))   # TODO
            res = session.execute(text_obj).fetchone()
            res = res[0] if res else None
            print('res =', res)  # TODO
        except IntegrityError as e:
            return e.code
        except InternalError as e:
            return e.code
        else:
            return res

    @staticmethod
    def get_table_obj(session, text_obj):
        """
        represent table as list of dicts
        :param session:
        :param text_obj:
        :return: list of dicts
        """
        try:
            rows = session.execute(text_obj).fetchall()
            res = []
            # print(rows)
            for row in rows:
                # print('type(row) =', type(row))
                # print('dict(row) =', dict(row))
                res.append(dict(row))
        except IntegrityError as e:
            return e.code
        except InternalError as e:
            return e.code
        else:
            return res

    @staticmethod
    def get_row_as_dict(session, text_obj):
        """
        represent row as dict
        :param session:
        :param text_obj:
        :return: dict
        """
        try:
            row = session.execute(text_obj).fetchone()
            # print(row)
        except IntegrityError as e:
            return e.code
        except InternalError as e:
            return e.code
        else:
            return dict(row)

    @staticmethod
    def get_json(session, text_obj):
        """
        Handles requests that return one field containing a json response
        :return:
        """
        try:
            print('text_obj.compile =', text_obj.compile(dialect=postgresql.dialect()))  # TODO
            res = session.execute(text_obj).fetchone()
            rows = res[0] if res is not None and res[0] is not None else []
            rows_list = []
            for row in rows:
                rows_list.append(dict(row))
        except IntegrityError as e:
            return e.code
        except InternalError as e:
            return e.code
        else:
            return rows_list

    @staticmethod
    def rows_to_dict(rows):
        rows_list = []
        for row in rows:
            rows_list.append(dict(row))
        return rows_list

    @staticmethod
    def type_converter(row_dict):
        for item in row_dict:

            if isinstance(row_dict[item], datetime):
                str_time = str(row_dict[item].strftime('%Y-%m-%d %H:%M:%S'))
                result = str(datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S').replace(tzinfo=pytz.utc))
                row_dict[item] = result
            if isinstance(row_dict[item], Decimal):
                row_dict[item] = float(row_dict[item])
        return row_dict

    @staticmethod
    def show_exceptions(fn):
        ex_type, ex_value, ex_trace = fn
        print("Type: ", ex_type)
        print("Value:", ex_value)
        print("Trace:", ex_trace)
        print("\n", "print_exception() ".center(40, "-"))
        traceback.print_exception(ex_type, ex_value, ex_trace, limit=5,
                                  file=sys.stdout)
        print("\n", "print_tb()".center(40, "-"))
        traceback.print_tb(ex_trace, limit=1, file=sys.stdout)
        print("\n", "format_exception()".center(40, "-"))
        print(traceback.format_exception(ex_type, ex_value, ex_trace, limit=5))
        print("\n", "format _ exception _ only () ".center(40, "-"))
        print(traceback.format_exception_only(ex_type, ex_value))
