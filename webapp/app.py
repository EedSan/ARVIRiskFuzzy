from flask import Flask, render_template, request, jsonify
import datetime
import objc
from Foundation import NSObject
from EventKit import EKEventStore, EKEvent, EKCalendar, EKSource, EKAuthorizationStatusAuthorized, EKEntityTypeEvent

app = Flask(__name__)


def get_event_store():
    event_store = EKEventStore.alloc().init()
    auth_status = event_store.authorizationStatusForEntityType_(EKEntityTypeEvent)

    if auth_status != EKAuthorizationStatusAuthorized:
        def requestAccessCompletionHandler(granted, error):
            if not granted:
                raise Exception("Access to Calendar not granted")

        event_store.requestAccessToEntityType_completion_(EKEntityTypeEvent, requestAccessCompletionHandler)

    return event_store


def create_calendar(event_store, calendar_name="TestAppCalendar"):
    # Check if the calendar already exists
    for calendar in event_store.calendarsForEntityType_(EKEntityTypeEvent):
        if calendar.title() == calendar_name:
            return calendar

    # If not, create a new one
    source = None
    for s in event_store.sources():
        if s.sourceType() == 1:  # 1 is for Local Source
            source = s
            break

    new_calendar = EKCalendar.calendarForEntityType_eventStore_(EKEntityTypeEvent, event_store)
    new_calendar.setTitle_(calendar_name)
    new_calendar.setSource_(source)

    event_store.saveCalendar_commit_error_(new_calendar, True, None)

    return new_calendar


def add_event_to_calendar(session_name, start_time, end_time):
    event_store = get_event_store()
    calendar = create_calendar(event_store)

    new_event = EKEvent.eventWithEventStore_(event_store)
    new_event.setTitle_(session_name)
    new_event.setStartDate_(start_time)
    new_event.setEndDate_(end_time)
    new_event.setCalendar_(calendar)

    event_store.saveEvent_span_error_(new_event, 0, None)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start_session', methods=['POST'])
def start_session():
    session_name = request.json['session_name']
    start_time = datetime.datetime.now()
    end_time = start_time + datetime.timedelta(minutes=25)

    # Add session to Apple Calendar
    add_event_to_calendar(session_name, start_time, end_time)

    return jsonify({"message": "Session started", "end_time": end_time.strftime("%H:%M:%S")})


@app.route('/pause_session', methods=['POST'])
def pause_session():
    return jsonify({"message": "Session paused"})


@app.route('/stop_session', methods=['POST'])
def stop_session():
    return jsonify({"message": "Session stopped"})


if __name__ == "__main__":
    app.run(debug=True)
