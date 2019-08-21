import React, { PureComponent } from 'react';
import 'react-dates/initialize';
import 'react-dates/lib/css/_datepicker.css';
import moment from 'moment';
import { DateRangePicker, isInclusivelyAfterDay } from 'react-dates';
import './Calendar.css';

export default class Calendar extends PureComponent {
  render() {
    const { startDate, endDate, displayFormat = 'DD MMM YYYY', focusedInput, onDatesChange, onFocusChange } = this.props;

    return (
      <div className="calendar_wrapper">
        <div className="calendar_inner">
          <div className="calendar_icon">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
              <line x1="16" y1="2" x2="16" y2="6" />
              <line x1="8" y1="2" x2="8" y2="6" />
              <line x1="3" y1="10" x2="21" y2="10" />
            </svg>
          </div>
          <div className="calendar_inputholder">
            <DateRangePicker
              startDate={startDate}
              startDateId="startDate"
              endDate={endDate}
              endDateId="endDate"
              displayFormat={displayFormat}
              minimumNights={0}
              isOutsideRange={day =>
                isInclusivelyAfterDay(day, moment.utc().add(1, 'days'))
              }
              onDatesChange={onDatesChange}
              focusedInput={focusedInput}
              onFocusChange={onFocusChange}
            />
          </div>
        </div>
      </div>
    );
  }
}
