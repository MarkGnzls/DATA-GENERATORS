/* main.js — unified + fixed + guaranteed working */

$(function () {

  console.log("MAIN JS LOADED");  // Debugging

  // ---------------------------------------------------
  // Utility Helpers
  // ---------------------------------------------------
  function showLoading() { $("#loadingOverlay").removeClass("d-none"); }
  function hideLoading() { $("#loadingOverlay").addClass("d-none"); }
  function toast(target, type, msg) {
    $(target).html(`<div class="alert alert-${type}">${msg}</div>`);
  }
  function clearAlerts() { $(".alert").remove(); }

  // ---------------------------------------------------
  // Sidebar Toggle (Mobile)
  // ---------------------------------------------------
  $("#sidebarToggle").on("click", function () {
    $("#sidebarMenu").toggleClass("open");
    $("body").toggleClass("overlay-open", $("#sidebarMenu").hasClass("open"));
  });

  $(document).on("click", function (e) {
    if (window.innerWidth < 992 && $("#sidebarMenu").hasClass("open")) {
      if (!$(e.target).closest("#sidebarMenu, #sidebarToggle").length) {
        $("#sidebarMenu").removeClass("open");
        $("body").removeClass("overlay-open");
      }
    }
  });

  $(window).on("resize", function () {
    if (window.innerWidth >= 992) {
      $("#sidebarMenu").removeClass("open");
      $("body").removeClass("overlay-open");
    }
  });

  // ---------------------------------------------------
  // HOME PAGE — Quick Actions
  // ---------------------------------------------------
  $("#quickGenerateBtn").on("click", function () {
    console.log("QuickGenerate clicked");
    window.location.href = "/dataset_builder";
  });

  $("#quickTrainBtn").on("click", function () {
    console.log("QuickTrain clicked");
    window.location.href = "/model_training";
  });

  $("#openDatasetBuilder").on("click", function () {
    console.log("OpenDatasetBuilder clicked");
    window.location.href = "/dataset_builder";
  });

  // ---------------------------------------------------
  // DATASET BUILDER
  // ---------------------------------------------------

  /* ===== Unified Dataset Builder + Model Training handlers =====
   Drop this into static/js/main.js (replace existing dataset/train sections).
   It supports multiple ID variants to be robust with your templates.
*/

  // small helpers
  function showLoading(){ $("#loadingOverlay").removeClass("d-none"); }
  function hideLoading(){ $("#loadingOverlay").addClass("d-none"); }
  function toast(target, type, html){ $(target).html(`<div class="alert alert-${type}">${html}</div>`); }

  // --- Utility: build class param inputs (works for #classNames or #classNamesInput)
  function buildClassSettingsInput(selectorClassNames="#classNames, #classNamesInput"){
    $(selectorClassNames).each(function(){
      const classes = $(this).val() ? $(this).val().split(",").map(s=>s.trim()).filter(Boolean) : [];
      let html = "";
      classes.forEach(c=>{
        html += `<div class="card p-2 mt-2">
                  <h6 class="mb-2">${c} Settings</h6>
                  <label class="form-label">Mean</label>
                  <input data-class="${c}" class="form-control class-mean" value="100">
                  <label class="form-label mt-2">Std Dev</label>
                  <input data-class="${c}" class="form-control class-std" value="10">
                 </div>`;
      });
      // write to either #classSettings or #classSettingsArea if present
      if ($("#classSettings").length) $("#classSettings").html(html || `<div class="text-muted small">Add classes separated by commas.</div>`);
      if ($("#classSettingsArea").length) $("#classSettingsArea").html(html || `<div class="text-muted small">Add classes separated by commas.</div>`);
    });
  }

  // Bind inputs to update
  $(document).on("input", "#classNames, #classNamesInput", function(){ buildClassSettingsInput(); });
  buildClassSettingsInput(); // initial

  // Keep sampleCount label updated (supports both IDs)
  $(document).on("input change", "#sampleCount, #sampleCountInput", function(){
    if ($("#sampleCountLabel").length) $("#sampleCountLabel").text($(this).val());
    if ($("#sampleCountLabelAlt").length) $("#sampleCountLabelAlt").text($(this).val());
  });

  // ---------- GENERATE DATASET (works if button ID is #btnGenerateDataset or #generateDataset or #generateSynthetic) ----------
  async function generateDatasetHandler(){
    $("#dsAlerts").html(""); if ($("#edaPreviewArea").length) $("#edaPreviewArea").html("");

    // gather features and classes from possible inputs
    const features = ($("#featureNames").val() || $("#featureNamesInput").val() || "").trim();
    const classNamesStr = ($("#classNames").val() || $("#classNamesInput").val() || "").trim();
    const samples = Number($("#sampleCount").val() || $("#sampleCountInput").val() || 5000);
    const seed = Number($("#seed").val() || 42);

    if (!features) return toast("#dsAlerts","warning","Enter feature names (comma-separated).");
    if (!classNamesStr) return toast("#dsAlerts","warning","Enter class names (comma-separated).");

    // collect class params
    const classParams = [];
    classNamesStr.split(",").map(s=>s.trim()).filter(Boolean).forEach(c=>{
      const mean = Number($(`.class-mean[data-class="${c}"]`).val() || 100);
      const std = Number($(`.class-std[data-class="${c}"]`).val() || 10);
      classParams.push({name: c, mean, std});
    });

    showLoading();
    try {
      const resp = await fetch("/api/generate_custom", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ features: features, class_params: classParams, samples: samples, seed: seed })
      });
      const data = await resp.json();
      if (data.status !== "ok") throw data.message || "Generation failed";

      toast("#dsAlerts","success", `Dataset generated — <a href="${data.csv_url}" target="_blank">Open CSV</a>`);
      // set the CSV URL to training CSV input if exists
      if ($("#trainCsvUrl").length) $("#trainCsvUrl").val(data.csv_url);
      if ($("#trainCsvUrlAlt").length) $("#trainCsvUrlAlt").val(data.csv_url);

      // request EDA (if preview area present)
      if ($("#edaPreviewArea").length){
        const edaResp = await fetch("/api/eda", {
          method: "POST", headers: {"Content-Type":"application/json"},
          body: JSON.stringify({ csv_url: data.csv_url })
        });
        const eda = await edaResp.json();
        if (eda.status === "ok" && eda.images){
          let out = "";
          if (eda.images.class_count) out += `<img class="eda-img" src="data:image/png;base64,${eda.images.class_count}">`;
          if (eda.images.histograms) out += `<img class="eda-img" src="data:image/png;base64,${eda.images.histograms}">`;
          if (eda.images.heatmap) out += `<img class="eda-img" src="data:image/png;base64,${eda.images.heatmap}">`;
          $("#edaPreviewArea").html(out);
        }
      }

    } catch (err){
      console.error(err);
      toast("#dsAlerts","danger", String(err));
    } finally { hideLoading(); }
  }

  // wire the generate handlers for any of the button ids you may have
  $(document).on("click", "#btnGenerateDataset, #generateDataset, #generateSynthetic", function(e){
    e.preventDefault(); generateDatasetHandler();
  });

  // ---------- TRAINING: wire Start Training (works for #btnStartTraining or #startTrain or #startTrainBtn) ----------
  async function startTrainingHandler(){
    $("#trainAlerts").html("");
    $("#splitInfo").text("Processing...");
    $("#classReport").text("");
    $("#confusionMatrixArea").html("");
    $("#originalSample").html("");
    $("#scaledSample").html("");
    $("#metricsTable").html("");
    $("#bestModelSummary").html("");
    $("#modelDownload").html("");

    const payload = {
      csv_url: ($("#trainCsvUrl").val() || $("#trainCsvUrlAlt").val() || "").trim(),
      features: ($("#trainFeatures").val() || $("#trainFeaturesAlt").val() || "").trim(),
      target: ($("#trainTarget").val() || $("#trainTargetAlt").val() || "").trim(),
      test_split: Number($("#trainTestSplit").val() || 30),
      kfold: Number($("#trainKFold").val() || 0)
    };

    if (!payload.csv_url) return toast("#trainAlerts","warning","CSV URL required (use Dataset Builder).");
    if (!payload.features) return toast("#trainAlerts","warning","Enter features (comma-separated).");
    if (!payload.target) return toast("#trainAlerts","warning","Enter target column name.");

    showLoading();
    try {
      const resp = await fetch("/api/train", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify(payload)
      });
      const data = await resp.json();
      if (data.status !== "ok") {
        toast("#trainAlerts","danger", data.message || "Training failed");
        $("#splitInfo").text("Training failed.");
        return;
      }

      // show split
      const si = data.split_info || {};
      if ($("#splitInfo").length){
        $("#splitInfo").html(`<strong>Total:</strong> ${si.total} &nbsp; <strong>Train:</strong> ${si.train} (${si.train_pct}) &nbsp; <strong>Test:</strong> ${si.test} (${si.test_pct})`);
      }

      // best model summary
      if ($("#bestModelSummary").length){
        $("#bestModelSummary").html(`<div><strong>Best Model:</strong> ${data.best_model} — <strong>Accuracy:</strong> ${(data.best_accuracy||0).toFixed(4)}</div>`);
      }

      // metrics table
      if ($("#metricsTable").length && data.metrics){
        let table = `<table class="table table-sm"><thead><tr><th>Model</th><th>Accuracy</th><th>F1</th></tr></thead><tbody>`;
        Object.entries(data.metrics).forEach(([m,val])=>{
          if (val.error) table += `<tr><td>${m}</td><td colspan="2" class="text-danger">${val.error}</td></tr>`;
          else table += `<tr><td>${m}</td><td>${(val.accuracy||0).toFixed(4)}</td><td>${(val.f1||0).toFixed(4)}</td></tr>`;
        });
        table += `</tbody></table>`;
        $("#metricsTable").html(table);
      }

      // class report
      if ($("#classReport").length) $("#classReport").text(data.classification_report || "No report");

      // confusion matrix
      if (data.confusion_matrix_img && $("#confusionMatrixArea").length) {
        $("#confusionMatrixArea").html(`<img class="img-fluid border" src="data:image/png;base64,${data.confusion_matrix_img}">`);
      }

      // samples
      function sampleTable(arr){
        if (!arr || !arr.length) return "<div class='text-muted small'>No sample</div>";
        let html = "<div class='table-responsive'><table class='table table-sm small'><thead><tr>";
        Object.keys(arr[0]).forEach(h => html += `<th>${h}</th>`);
        html += "</tr></thead><tbody>";
        arr.forEach(r => { html += "<tr>"; Object.values(r).forEach(v => html += `<td>${v}</td>`); html += "</tr>"; });
        html += "</tbody></table></div>";
        return html;
      }
      if ($("#originalSample").length) $("#originalSample").html(sampleTable(data.original_sample));
      if ($("#scaledSample").length) $("#scaledSample").html(sampleTable(data.scaled_sample));

      if (data.model_url){
        if ($("#modelDownload").length) $("#modelDownload").html(`<a class="btn btn-outline-secondary btn-sm" href="${data.model_url}" target="_blank">Download Best Model</a>`);
        if ($("#predictModelUrl").length) $("#predictModelUrl").val(data.model_url);
      }

    } catch(err){
      console.error(err); toast("#trainAlerts","danger", String(err));
    } finally { hideLoading(); }
  }

  $(document).on("click", "#btnStartTraining, #startTrain, #startTrainBtn", function(e){ e.preventDefault(); startTrainingHandler(); });

  // ---------- Upload model (works for multiple upload button IDs) ----------
  $(document).on("click", "#btnUploadModel, #uploadModelBtn", async function(e){
    e.preventDefault();
    $("#uploadStatus").html("");
    const file = ($("#uploadModelFile")[0] || {}).files ? $("#uploadModelFile")[0].files[0] : ($("#modelFileUpload")[0]||{}).files[0];
    const features = ($("#uploadModelFeatures").val() || "").trim();
    if (!file) return toast("#uploadStatus","warning","Choose a model file.");
    const fd = new FormData(); fd.append("model_file", file); fd.append("features", features);
    showLoading();
    try {
      const resp = await fetch("/api/upload_model", { method:"POST", body: fd });
      const data = await resp.json();
      if (data.status !== "ok") throw data.message || "Upload failed";
      toast("#uploadStatus","success","Model uploaded. Model URL set for prediction.");
      if ($("#predictModelUrl").length) $("#predictModelUrl").val(data.model_url);
    } catch(err){ toast("#uploadStatus","danger", String(err)); }
    finally{ hideLoading(); }
  });

  // ---------- Prediction (works if prediction button id varies) ----------
  function buildPredictInputsFromFeatures(){
    const featStr = ($("#trainFeatures").val() || $("#uploadModelFeatures").val() || "").trim();
    const features = featStr ? featStr.split(",").map(s=>s.trim()).filter(Boolean) : [];
    if (!features.length) { if ($("#predictFeatureInputs").length) $("#predictFeatureInputs").html("<div class='text-muted small'>Enter features to build inputs.</div>"); return; }
    let html = "<div class='row g-2'>";
    features.forEach(f => html += `<div class='col-md-6'><label class='form-label'>${f}</label><input class='form-control predict-value' data-feature='${f}'></div>`);
    html += "</div>";
    $("#predictFeatureInputs").html(html);
  }
  $(document).on("input", "#trainFeatures, #uploadModelFeatures", buildPredictInputsFromFeatures);
  buildPredictInputsFromFeatures();

  $(document).on("click", "#btnMakePrediction, #makePrediction, #predictBtn", async function(e){
    e.preventDefault();
    $("#predictionResult").html("");
    const model_url = ($("#predictModelUrl").val() || "").trim();
    if (!model_url) return toast("#predictionResult","warning","Provide model URL.");
    const features_map = {};
    let missing = false;
    $(".predict-value").each(function(){ const k=$(this).data("feature"); const v=$(this).val(); if (v===undefined||v==="") missing=true; features_map[k]=v; });
    if (missing) return toast("#predictionResult","warning","Fill all feature inputs.");
    showLoading();
    try {
      const resp = await fetch("/api/predict", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({ model_url, features_map })});
      const data = await resp.json();
      if (data.status !== "ok") throw data.message || "Prediction failed";
      let out = `<div class="alert alert-success"><strong>Prediction:</strong> ${data.prediction}</div>`;
      if (data.probabilities) out += `<pre class="small bg-light p-2">${JSON.stringify(data.probabilities,null,2)}</pre>`;
      $("#predictionResult").html(out);
    } catch(err){
      toast("#predictionResult","danger", String(err));
    } finally { hideLoading(); }
  });


  /* -------------------------
   MODEL TRAINING — FINAL MERGED VERSION
--------------------------*/

$("#trainTestSplit").on("input change", function () {
    $("#trainTestLabel").text($(this).val() + "%");
});

$("#btnStartTraining").on("click", async function () {
    $("#trainAlerts").html("");
    $("#splitInfo").text("Processing...");
    $("#classReport").text("");
    $("#confusionMatrixArea").html("");
    $("#originalSample").html("");
    $("#scaledSample").html("");
    $("#metricsTable").html("");
    $("#bestModelSummary").html("");
    $("#modelDownload").html("");

    const payload = {
        csv_url: $("#trainCsvUrl").val().trim(),
        features: $("#trainFeatures").val().trim(),
        target: $("#trainTarget").val().trim(),
        test_split: Number($("#trainTestSplit").val() || 30),
        kfold: Number($("#trainKFold").val() || 0)
    };

    if (!payload.csv_url)
        return toast("#trainAlerts", "warning", "CSV URL is required.");
    if (!payload.features)
        return toast("#trainAlerts", "warning", "Enter feature names.");
    if (!payload.target)
        return toast("#trainAlerts", "warning", "Enter target column name.");

    showLoading();
    try {
        const resp = await fetch("/api/train", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await resp.json();
        if (data.status !== "ok") {
            toast("#trainAlerts", "danger", data.message || "Training failed");
            $("#splitInfo").text("Training failed.");
            return;
        }

        // Split info
        const si = data.split_info || {};
        $("#splitInfo").html(`
            <strong>Total:</strong> ${si.total} —
            <strong>Train:</strong> ${si.train} (${si.train_pct}) —
            <strong>Test:</strong> ${si.test} (${si.test_pct})
        `);

        // Best model summary
        $("#bestModelSummary").html(`
            <div><strong>Best Model:</strong> ${data.best_model}
            — <strong>Accuracy:</strong> ${(data.best_accuracy || 0).toFixed(4)}</div>
        `);

        // Metrics comparison table
        let table = `<table class="table table-sm"><thead>
                        <tr><th>Model</th><th>Accuracy</th><th>F1 Score</th></tr>
                     </thead><tbody>`;
        Object.entries(data.metrics).forEach(([m, val]) => {
            if (val.error)
                table += `<tr><td>${m}</td><td colspan="2" class="text-danger">${val.error}</td></tr>`;
            else
                table += `<tr><td>${m}</td><td>${val.accuracy.toFixed(4)}</td><td>${val.f1.toFixed(4)}</td></tr>`;
        });
        table += "</tbody></table>";
        $("#metricsTable").html(table);

        // Classification report
        $("#classReport").text(data.classification_report || "No report");

        // Confusion matrix image
        if (data.confusion_matrix_img)
            $("#confusionMatrixArea").html(
                `<img class="img-fluid border" src="data:image/png;base64,${data.confusion_matrix_img}">`
            );

        // Samples
        function sampleTable(arr) {
            if (!arr || arr.length === 0) return "<div class='text-muted small'>No sample</div>";
            let html = "<table class='table table-sm small'><thead><tr>";
            Object.keys(arr[0]).forEach(k => html += `<th>${k}</th>`);
            html += "</tr></thead><tbody>";
            arr.forEach(r => {
                html += "<tr>";
                Object.values(r).forEach(v => html += `<td>${v}</td>`);
                html += "</tr>";
            });
            html += "</tbody></table>";
            return html;
        }

        $("#originalSample").html(sampleTable(data.original_sample));
        $("#scaledSample").html(sampleTable(data.scaled_sample));

        // Model URL
        if (data.model_url) {
            $("#predictModelUrl").val(data.model_url);
            $("#modelDownload").html(`
                <a href="${data.model_url}" target="_blank" class="btn btn-outline-secondary btn-sm">Download Best Model</a>
            `);
        }

    } catch (err) {
        console.error(err);
        toast("#trainAlerts", "danger", String(err));
    } finally {
        hideLoading();
    }
});

  // ---------------------------------------------------
  // MODEL UPLOAD
  // ---------------------------------------------------
  $("#btnUploadModel").on("click", async function () {
    const file = $("#uploadModelFile")[0].files[0];
    const features = $("#uploadModelFeatures").val().trim();

    if (!file) return toast("#uploadStatus", "warning", "Choose a model file.");

    const fd = new FormData();
    fd.append("model_file", file);
    fd.append("features", features);

    showLoading();
    try {
      const resp = await fetch("/api/upload_model", { method: "POST", body: fd });
      const data = await resp.json();
      if (data.status !== "ok") throw data.message;

      toast("#uploadStatus", "success", "Model uploaded successfully.");
      $("#predictModelUrl").val(data.model_url);

    } catch (err) {
      toast("#uploadStatus", "danger", err);
    } finally {
      hideLoading();
    }
  });

  // ---------------------------------------------------
  // PREDICTION
  // ---------------------------------------------------
  function buildPredictInputs() {
    const features = $("#trainFeatures").val().split(",").map(s => s.trim()).filter(Boolean);
    let html = "";

    features.forEach(f => {
      html += `
        <div class="col-md-6">
          <label>${f}</label>
          <input class="form-control predict-value" data-feature="${f}">
        </div>`;
    });

    $("#predictFeatureInputs").html(`<div class="row g-2">${html}</div>`);
  }

  $("#trainFeatures").on("input", buildPredictInputs);
  buildPredictInputs();

  $("#btnMakePrediction").on("click", async function () {
    const model_url = $("#predictModelUrl").val().trim();
    if (!model_url) return toast("#predictionResult", "warning", "Enter model URL.");

    const values = {};
    $(".predict-value").each(function () {
      values[$(this).data("feature")] = $(this).val();
    });

    showLoading();
    try {
      const resp = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_url, features_map: values })
      });

      const data = await resp.json();
      if (data.status !== "ok") throw data.message;

      $("#predictionResult").html(`
        <div class="alert alert-success">
          <strong>Prediction:</strong> ${data.prediction}
        </div>
      `);

    } catch (err) {
      toast("#predictionResult", "danger", err);
    } finally {
      hideLoading();
    }
  });

  // ---------------------------------------------------
  // ALGO GUIDE
  // ---------------------------------------------------
  const ALGO_DEFS = {
    LogisticRegression: { title:"Logistic Regression", desc:"Linear model for classification using the logistic function." },
    RandomForest: { title:"Random Forest", desc:"Ensemble of decision trees." },
    GradientBoosting: { title:"Gradient Boosting", desc:"Boosting ensemble." },
    SVC: { title:"Support Vector Classifier", desc:"Margin-based classifier." },
    KNN: { title:"K-Nearest Neighbors", desc:"Instance-based classifier." },
    DecisionTree: { title:"Decision Tree", desc:"Interpretable tree-based model." }
};

$("#algoSelect").on("change", function(){
    const info = ALGO_DEFS[$(this).val()] || {};
    $("#algoDesc").html(`<strong>${info.title || ""}</strong><p class="small text-muted">${info.desc || ""}</p>`);
});


}); // END READY
