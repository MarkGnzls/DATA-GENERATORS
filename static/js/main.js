/* =====================================================
   main.js — CLEAN, ALIGNED, FINAL VERSION (FIXED)
   ===================================================== */

$(function () {

  console.log("MAIN JS LOADED");

  /* ===============================
     GLOBAL STATE
  =============================== */
  let CURRENT_CSV = null;
  let CURRENT_MODEL = null;

  /* ===============================
     HELPERS
  =============================== */
  function showLoading() {
    $("#loadingOverlay").removeClass("d-none");
  }

  function hideLoading() {
    $("#loadingOverlay").addClass("d-none");
  }

  function toast(target, type, msg) {
    $(target).html(`<div class="alert alert-${type}">${msg}</div>`);
  }

  /* ===============================
     HOME PAGE LOGIC (FIXED)
  =============================== */
  const isHome =
    $("#quickGenerateBtn").length ||
    $("#quickTrainBtn").length ||
    $("#openDatasetBuilder").length;

  if (isHome) {

    const summary = localStorage.getItem("latestRunSummary");

    if (summary) {
      const s = JSON.parse(summary);
      let badge = "secondary";
      let icon = "bi-info-circle";

      if ((s.action || "").includes("Dataset")) {
        badge = "primary";
        icon = "bi-database";
      } else if ((s.action || "").includes("Model")) {
        badge = "success";
        icon = "bi-cpu";
      }

      $("#summaryContent").html(`
        <div class="mb-2">
          <span class="badge bg-${badge}">
            <i class="bi ${icon} me-1"></i>${s.action}
          </span>
        </div>
        <ul class="small mb-2">
          ${s.rows ? `<li><strong>Rows:</strong> ${s.rows}</li>` : ""}
          ${s.features ? `<li><strong>Features:</strong> ${s.features.join(", ")}</li>` : ""}
          ${s.classes ? `<li><strong>Classes:</strong> ${s.classes.join(", ")}</li>` : ""}
        </ul>
        <p class="small text-muted"><i class="bi bi-clock"></i> ${s.time}</p>
        ${s.csv_url ? `<a href="${s.csv_url}" target="_blank"><i class="bi bi-download"></i> Download CSV</a>` : ""}
      `);
    }

  }

  
  /* ===============================
     DATASET BUILDER
  =============================== */
  if ($("#btnGenerateDataset").length) {

    $("#btnGenerateDataset").on("click", async function () {

      const features = $("#featureNames").val().trim();
      const classNames = $("#classNames").val().trim();

      if (!features || !classNames)
        return toast("#dsAlerts", "warning", "Missing fields.");

      const class_params = classNames.split(",").map(cls => ({
        name: cls.trim(),
        mean: Number($(`.mean-input[data-class="${cls.trim()}"]`).val() || 50),
        std: Number($(`.std-input[data-class="${cls.trim()}"]`).val() || 10)
      }));

      showLoading();
      try {
        const resp = await fetch("/api/generate_custom", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            features,
            class_params,
            samples: Number($("#sampleCount").val()),
            seed: Number($("#seed").val())
          })
        });

        const data = await resp.json();
        if (data.status !== "ok") throw data.message;

        CURRENT_CSV = data.csv_url;
        $("#datasetPreview").html(`<a href="${data.csv_url}" target="_blank">Download CSV</a>`);
        runEDA(CURRENT_CSV);

      } catch (e) {
        toast("#dsAlerts", "danger", e);
      } finally {
        hideLoading();
      }
    });
  }

  /* ===============================
     EDA
  =============================== */
  async function runEDA(csv_url) {
    $("#edaPreviewArea").html("Running EDA...");
    const resp = await fetch("/api/eda", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ csv_url })
    });
    const data = await resp.json();
    if (data.status !== "ok") return;

    let html = "";
    Object.entries(data.images || {}).forEach(([k, img]) => {
      html += `
        <div class="col-md-6">
          <img class="img-fluid border" src="data:image/png;base64,${img}">
        </div>`;
    });

    $("#edaPreviewArea").html(html);
  }

  /* ===============================
     MODEL TRAINING (FIXED)
  =============================== */
  if ($("#btnStartTraining").length) {

    $("#btnStartTraining").off("click.training").on("click.training", async function () {

      const payload = {
        csv_url: $("#trainCsvUrl").val().trim(),
        features: $("#trainFeatures").val().trim(),
        target: $("#trainTarget").val().trim(),
        test_split: Number($("#trainTestSplit").val() || 30),
        kfold: Number($("#trainKFold").val() || 0)
      };

      if (!payload.csv_url || !payload.features || !payload.target)
        return toast("#trainAlerts", "warning", "Missing fields.");

      showLoading();
      const resp = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await resp.json();
      hideLoading();

      if (data.status !== "ok") return;

      CURRENT_MODEL = data.model_url;
      $("#predictModelUrl").val(CURRENT_MODEL);
      $("#classReport").text(data.classification_report || "");
    });
  }

  /* ===============================
     PREDICTION
  =============================== */
  if ($("#btnMakePrediction").length) {

    $("#btnMakePrediction").on("click", async function () {

      const features_map = {};
      $(".predict-value").each(function () {
        features_map[$(this).data("feature")] = $(this).val();
      });

      const resp = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_url: $("#predictModelUrl").val(),
          features_map
        })
      });

      const data = await resp.json();
      $("#predictionResult").html(`<strong>Prediction:</strong> ${data.prediction}`);
    });
  }
  /* ===============================
   RESTORE STATE ON PAGE LOAD
=============================== */
if (
  $("#homeSummary").length ||
  $("#datasetPreview").length ||
  $("#edaPreviewArea").length
) {
  fetch("/api/state")
    .then(res => res.json())
    .then(state => {
      if (!state || !state.action) return;

      /* ---------- HOME PAGE ---------- */
      if ($("#homeSummary").length) {
        $("#summaryContent").html(`
          <div class="mb-2">
            <span class="badge bg-primary">
              <i class="bi bi-database me-1"></i>${state.action}
            </span>
          </div>
          <ul class="small">
            <li><strong>Rows:</strong> ${state.rows}</li>
            <li><strong>Features:</strong> ${state.features.join(", ")}</li>
            <li><strong>Classes:</strong> ${state.classes.join(", ")}</li>
          </ul>
          <a href="${state.csv_url}" target="_blank">Download CSV</a>
        `);
      }

      /* ---------- DATASET BUILDER ---------- */
      if ($("#datasetPreview").length) {
        $("#datasetPreview").html(`
          <strong>${state.rows}</strong> rows generated<br>
          <a href="${state.csv_url}" target="_blank">Download CSV</a>
        `);
      }

      /* ---------- AUTO-RUN EDA AGAIN ---------- */
      if (typeof runEDA === "function" && $("#edaPreviewArea").length) {
        runEDA(state.csv_url);
      }
    })
    .catch(err => console.error("State restore failed:", err));
}

/* ===============================
   HOME — QUICK ACTIONS (FINAL)
=============================== */

if (
  $("#quickGenerateBtn").length ||
  $("#quickTrainBtn").length ||
  $("#openDatasetBuilder").length
) {

  // Generate Example Dataset
  $(document).on("click", "#quickGenerateBtn", async function () {
    console.log("Generate Example Dataset clicked");

    $("#summaryContent").html(
      "<p class='text-muted'>Generating example dataset...</p>"
    );

    try {
      const resp = await fetch("/api/generate_custom", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          features: "length,width,density,pH",
          class_params: [
            { name: "Ampalaya", mean: 50, std: 10 },
            { name: "Banana", mean: 65, std: 12 },
            { name: "Cabbage", mean: 40, std: 8 }
          ],
          samples: 3000,
          seed: 42
        })
      });

      const data = await resp.json();
      if (data.status !== "ok") throw data.message;

      localStorage.setItem("latestRunSummary", JSON.stringify({
        action: "Example Dataset Generated",
        rows: data.n,
        features: ["length", "width", "density", "pH"],
        classes: ["Ampalaya", "Banana", "Cabbage"],
        csv_url: data.csv_url,
        time: new Date().toLocaleString()
      }));

      location.reload();

    } catch (err) {
      console.error(err);
      $("#summaryContent").html(
        "<div class='alert alert-danger'>Failed to generate dataset.</div>"
      );
    }
  });

  // Go to Model Training
  $(document).on("click", "#quickTrainBtn", function () {
    console.log("Go to Model Training clicked");
    window.location.href = "/model_training";
  });

  // Open Dataset Builder
  $(document).on("click", "#openDatasetBuilder", function () {
    console.log("Open Dataset Builder clicked");
    window.location.href = "/dataset_builder";
  });
}

  

});
