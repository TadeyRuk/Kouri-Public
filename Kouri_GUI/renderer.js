async function sendMessage() {
  const msg = document.getElementById("message").value;
  const outEl = document.getElementById("response");

  if (!msg.trim()) {
    outEl.innerText = "Please type something!";
    return;
  }

  outEl.innerText = "Kouri is thinking…";

  try {
    const response = await fetch("http://127.0.0.1:8000/kouri", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: msg })
    });

    const data = await response.json();
    outEl.innerText = data.response ?? "(no reply)";
  } catch (err) {
    outEl.innerText = "Error: " + err.message;
  }
}
