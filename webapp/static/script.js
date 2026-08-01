let timer;
let isPaused = false;
let timeLeft = 1500;

function updateTimerDisplay() {
    const minutes = Math.floor(timeLeft / 60);
    const seconds = timeLeft % 60;
    document.getElementById("minutes").textContent = String(minutes).padStart(2, '0');
    document.getElementById("seconds").textContent = String(seconds).padStart(2, '0');
}

function startSession() {
    const sessionName = document.getElementById("sessionName").value || "Pomodoro Session";

    fetch('/start_session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_name: sessionName })
    }).then(response => response.json()).then(data => {
        console.log(data.message);
        timeLeft = 1500; // 25 minutes
        isPaused = false;
        if (!timer) {
            timer = setInterval(() => {
                if (!isPaused && timeLeft > 0) {
                    timeLeft--;
                    updateTimerDisplay();
                } else if (timeLeft <= 0) {
                    clearInterval(timer);
                    timer = null;
                    alert("Pomodoro session complete!");
                }
            }, 1000);
        }
    });
}

function pauseSession() {
    fetch('/pause_session', { method: 'POST' })
        .then(response => response.json())
        .then(data => console.log(data.message));

    isPaused = !isPaused;
}

function stopSession() {
    fetch('/stop_session', { method: 'POST' })
        .then(response => response.json())
        .then(data => console.log(data.message));

    clearInterval(timer);
    timer = null;
    timeLeft = 1500; // Reset to 25 minutes
    updateTimerDisplay();
}
