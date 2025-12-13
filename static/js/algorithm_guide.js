const algoData = {
  logreg: {
    desc: `
      <strong>Logistic Regression</strong> is a linear classification algorithm that
      models the probability of a class using the sigmoid function.
    `,
    formula: `
      $$ P(y=1|x) = \\frac{1}{1 + e^{-(w^Tx + b)}} $$
    `,
    fail: `
      Logistic Regression fails when data is <strong>non-linearly separable</strong>
      or when strong feature interactions exist.
    `,
    comparison: `
      <ul>
        <li>Fast training</li>
        <li>Interpretable</li>
        <li>Weak on complex patterns</li>
      </ul>
    `
  },

  knn: {
    desc: `
      <strong>KNN</strong> classifies data based on the majority class of nearest neighbors.
    `,
    formula: `
      $$ \\hat{y} = \\text{mode}(y_1, y_2, ..., y_k) $$
    `,
    fail: `
      KNN fails with <strong>high-dimensional data</strong> and large datasets.
    `,
    comparison: `
      <ul>
        <li>No training phase</li>
        <li>Slow prediction</li>
        <li>Sensitive to noise</li>
      </ul>
    `
  },

  tree: {
    desc: `
      <strong>Decision Trees</strong> split data using feature thresholds to form rules.
    `,
    formula: `
      $$ Gini = 1 - \\sum p_i^2 $$
    `,
    fail: `
      Decision Trees fail due to <strong>overfitting</strong> without pruning.
    `,
    comparison: `
      <ul>
        <li>Highly interpretable</li>
        <li>Handles non-linearity</li>
        <li>Unstable</li>
      </ul>
    `
  },

  svm: {
    desc: `
      <strong>SVM</strong> finds the optimal hyperplane maximizing class separation margin.
    `,
    formula: `
      $$ \\min ||w|| \\quad s.t. \\quad y(w^Tx + b) \\ge 1 $$
    `,
    fail: `
      SVM fails when <strong>classes overlap heavily</strong> or kernel is mischosen.
    `,
    comparison: `
      <ul>
        <li>Strong margins</li>
        <li>Kernel flexibility</li>
        <li>Hard to tune</li>
      </ul>
    `
  }
};

let chart;

function drawChart(type) {
  const ctx = document.getElementById("algoChart");

  if (chart) chart.destroy();

  chart = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Class A",
          data: Array.from({ length: 20 }, () => ({
            x: Math.random() * 4,
            y: Math.random() * 4
          })),
          backgroundColor: "blue"
        },
        {
          label: "Class B",
          data: Array.from({ length: 20 }, () => ({
            x: Math.random() * 4 + 3,
            y: Math.random() * 4 + 3
          })),
          backgroundColor: "red"
        }
      ]
    },
    options: {
      scales: {
        x: { title: { display: true, text: "Feature 1" } },
        y: { title: { display: true, text: "Feature 2" } }
      }
    }
  });
}

function updateAlgo(algo) {
  const d = algoData[algo];

  document.getElementById("algoDescription").innerHTML = d.desc;
  document.getElementById("algoFormula").innerHTML = d.formula;
  document.getElementById("algoFailure").innerHTML = d.fail;
  document.getElementById("algoComparison").innerHTML = d.comparison;

  MathJax.typeset();
  drawChart(algo);
}

document.getElementById("algoSelect").addEventListener("change", e => {
  updateAlgo(e.target.value);
});

updateAlgo("logreg");
