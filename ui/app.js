const form = document.getElementById("churn-form");
const resultDiv = document.getElementById("result");

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  resultDiv.textContent = ""; // clear previous text

  const formData = new FormData(form);
  const row = {};

  formData.forEach((value, key) => {
    const v = String(value).trim();

    // keep empty as empty (let backend reject if required)
    if (v === "") {
      row[key] = v;
      return;
    }

    // numeric fields come as strings; convert only if it's a real number
    const num = Number(v);
    row[key] = Number.isFinite(num) && v !== "" ? num : v;
  });

  const payload = { rows: [row], threshold: 0.5 };

  try {
    const response = await fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    // if backend returns 400/500, don't try to parse as success blindly
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `HTTP ${response.status}`);
    }

    const data = await response.json();

    // YOUR API SHAPE:
    const prob = data.probabilities?.[0];
    const label = data.predictions?.[0];

    if (prob === undefined || label === undefined) {
      throw new Error("Unexpected API response shape");
    }

    resultDiv.innerHTML =
      `Churn probability: ${Number(prob).toFixed(3)}<br>` +
      `Prediction: <strong>${label === 1 ? "Churn" : "No Churn"}</strong>`;
  } catch (err) {
    console.error(err);
    resultDiv.textContent = "Error connecting to API";
  }
});
