import React, { Component } from "react";
import "react-dates/initialize";
import "react-dates/lib/css/_datepicker.css";
import moment from "moment";
import { debounce } from "lodash";
import { DateRangePicker, isInclusivelyAfterDay } from "react-dates";

import "./Calendar.css";

export default class Calendar extends Component {
  state = {
    daySize: null,
    openDirection: null,
    numberOfMonths: null
  };

  componentDidMount() {
    this.handleResize();

    window.addEventListener("resize", debounce(this.handleResize, 300));
  }

  componentWillUnmount() {
    window.removeEventListener("resize", this.handleResize);
  }

  handleResize = () => {
    let state = {};

    window.innerWidth < 768 || window.innerHeight < 730
      ? (state = { ...state, numberOfMonths: 1, daySize: 32 })
      : (state = { ...state, numberOfMonths: 2, daySize: 38 });

    window.innerHeight < 690
      ? (state = { ...state, openDirection: "up" })
      : (state = { ...state, openDirection: "down" });

    this.setState(state);
  };

  render() {
    const { numberOfMonths, openDirection, daySize } = this.state;

    const {
      startDate,
      endDate,
      displayFormat = "DD MMM YYYY",
      focusedInput,
      onDatesChange,
      onFocusChange,
      onCalendarIconClick,
      onClose
    } = this.props;

    return (
      <div className="calendar_wrapper">
        <div className="calendar_inner">
          <div>
            <DateRangePicker
              hideKeyboardShortcutsPanel={true}
              numberOfMonths={numberOfMonths}
              openDirection={openDirection}
              daySize={daySize}
              startDate={startDate}
              startDateId="startDate"
              endDate={endDate}
              endDateId="endDate"
              displayFormat={displayFormat}
              minimumNights={0}
              isOutsideRange={day =>
                isInclusivelyAfterDay(day, moment.utc().add(1, "days"))
              }
              onDatesChange={onDatesChange}
              focusedInput={focusedInput}
              onFocusChange={onFocusChange}
              onClose={onClose}
            />
          </div>
          <div
            onClick={onCalendarIconClick}
            className={`calendar_icon${focusedInput ? " is-focused" : ""}`}
          >
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
        </div>
      </div>
    );
  }
}
