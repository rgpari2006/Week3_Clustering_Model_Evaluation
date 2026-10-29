const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, HeadingLevel,
  AlignmentType, WidthType, ShadingType, BorderStyle, LevelFormat, Footer, PageNumber, PageBreak,
} = require("docx");

const R = JSON.parse(fs.readFileSync("results.json", "utf8"));
const W = 9026; // A4 content width (DXA)
const FONT = "Calibri";
const f = (x, d = 3) => Number(x).toFixed(d);

const p = (text, opts = {}) =>
  new Paragraph({ spacing: { after: 120, line: 300 }, ...opts, children: Array.isArray(text) ? text : [new TextRun({ text, font: FONT, size: 22 })] });
const run = (t, o = {}) => new TextRun({ text: t, font: FONT, size: 22, ...o });
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT })] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT })] });
const bullet = (t) => new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 80 },
  children: Array.isArray(t) ? t : [run(t)] });
const caption = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
  children: [new TextRun({ text: t, font: FONT, size: 19, italics: true, color: "555555" })] });

function img(file, wPx, hPx) {
  const w = Math.min(wPx, 600), h = Math.round((hPx * w) / wPx);
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(`figures/${file}`), transformation: { width: w, height: h },
      altText: { title: file, description: file, name: file } })] });
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "BBBBBB" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(header, rows, widths) {
  const cell = (t, w, hd, align) => new TableCell({
    width: { size: w, type: WidthType.DXA }, borders,
    shading: hd ? { fill: "1F4E79", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 70, bottom: 70, left: 110, right: 110 },
    children: [new Paragraph({ alignment: align || AlignmentType.LEFT,
      children: [new TextRun({ text: String(t), font: FONT, size: 20, bold: hd, color: hd ? "FFFFFF" : "000000" })] })],
  });
  return new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((t, i) => cell(t, widths[i], true, i ? AlignmentType.CENTER : AlignmentType.LEFT)) }),
      ...rows.map((r) => new TableRow({ children: r.map((t, i) => cell(t, widths[i], false, i ? AlignmentType.CENTER : AlignmentType.LEFT)) })),
    ],
  });
}
const gap = () => new Paragraph({ spacing: { after: 160 }, children: [] });

const tm = R.test_metrics, lr = tm["Logistic Regression (tuned)"], rf = tm["Random Forest (tuned)"];
const cvLR = R.final_cv_full_data["Logistic Regression (tuned)"], cvRF = R.final_cv_full_data["Random Forest (tuned)"];

// crosstabs
const ct = (obj) => ["class_0", "class_1", "class_2"].map((c, i) => [`Cultivar ${i}`, obj["0"][c], obj["1"][c], obj["2"][c]]);

const children = [
  // ---- Title ----
  new Paragraph({ spacing: { before: 1200, after: 120 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Week 3 Task", font: FONT, size: 28, color: "1F4E79", bold: true })] }),
  new Paragraph({ spacing: { after: 200 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Unsupervised Learning & Model Evaluation", font: FONT, size: 52, bold: true, color: "1F4E79" })] }),
  new Paragraph({ spacing: { after: 120 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Clustering and Model Evaluation Report on the Wine Dataset", font: FONT, size: 28, color: "444444" })] }),
  new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: "1F4E79", space: 8 } }, spacing: { after: 300 }, children: [] }),
  p([run("Techniques: ", { bold: true }), run("K-Means, Hierarchical Clustering, PCA, Cross-Validation, Confusion Matrix, Precision, Recall, F1-score, ROC-AUC, GridSearchCV")], { alignment: AlignmentType.CENTER }),
  p([run("Tools: ", { bold: true }), run("Python, scikit-learn, pandas, NumPy, SciPy, Matplotlib, Seaborn")], { alignment: AlignmentType.CENTER }),
  new Paragraph({ children: [new PageBreak()] }),

  // ---- 1 ----
  h1("1. Introduction and Objectives"),
  p("This report covers the Week 3 task, whose goal is to learn clustering techniques and to evaluate machine learning models properly. It has three objectives: (1) understand clustering algorithms such as K-Means and Hierarchical Clustering together with dimensionality reduction; (2) learn cross-validation, the confusion matrix, precision, recall, F1-score, ROC-AUC and hyperparameter tuning; and (3) perform clustering on a dataset and prepare a model evaluation report."),
  p("The two halves of the task use different kinds of learning. Clustering is unsupervised and needs no labels, while precision, recall and ROC-AUC require known labels. To cover both in one coherent study, a labelled dataset is used. The clustering algorithms never see the labels; the labels are used afterwards to judge how meaningful the discovered clusters are, and to train and evaluate supervised classifiers."),

  // ---- 2 ----
  h1("2. Dataset"),
  p(`The Wine dataset from scikit-learn (UCI Machine Learning Repository) contains ${R.n_samples} wines grown in the same Italian region from three different cultivars. Each wine is described by ${R.n_features} continuous chemical measurements, such as alcohol, flavanoids, colour intensity, hue and proline. There are no missing values. The class distribution is ${R.class_counts.class_0} / ${R.class_counts.class_1} / ${R.class_counts.class_2} wines for cultivars 0, 1 and 2, which is mildly imbalanced, so stratified splitting and macro-averaged metrics are used throughout.`),
  p("All features are standardised (zero mean, unit variance) before clustering and PCA, because these methods are driven by distances and variance and would otherwise be dominated by large-scale features such as proline."),

  // ---- 3 ----
  h1("3. Dimensionality Reduction with PCA"),
  p(`Principal Component Analysis projects the data onto orthogonal directions of maximum variance. The first two components explain ${f(R.pca_var_2 * 100, 1)}% of the total variance, ${R.pca_comp_90} components are needed to retain 90%, and ${R.pca_comp_95} components retain 95%. Two components are therefore not enough to represent the data perfectly, but they are sufficient to visualise its cluster structure, and they are used for all 2-D plots in this report. Clustering itself was run on the full set of standardised features, not on the 2-D projection.`),
  img("01_pca_variance.png", 900, 600),
  caption("Figure 1. PCA explained variance (individual and cumulative)."),

  // ---- 4 ----
  h1("4. Clustering"),
  h2("4.1 K-Means and choosing k"),
  p("K-Means partitions the data into k clusters by minimising the within-cluster sum of squares (inertia). Because k must be chosen in advance, two complementary criteria were used for k = 2 to 10: the elbow method, which looks for the point where adding clusters stops reducing inertia sharply, and the silhouette score, which measures how well each point fits its own cluster compared with the nearest other cluster (higher is better, range −1 to 1)."),
  img("02_kmeans_elbow_silhouette.png", 1650, 600),
  caption("Figure 2. Elbow curve (left) and silhouette scores (right) for k = 2 to 10."),
  p(`The silhouette score peaks at k = ${R.best_k_by_silhouette} (${f(R.kmeans_silhouette_by_k["3"], 3)}), and the elbow curve bends at the same place, so k = 3 was selected. This agrees with the three known cultivars. The final K-Means model (k = 3, 10 initialisations) has a silhouette score of ${f(R.kmeans_k3_silhouette)} and an inertia of ${f(R.kmeans_k3_inertia, 1)}. A silhouette near 0.28 indicates moderate, somewhat overlapping clusters, which is typical for real chemical data.`),

  h2("4.2 Hierarchical clustering"),
  p("Agglomerative hierarchical clustering starts with every sample as its own cluster and repeatedly merges the two closest clusters. Ward linkage, which merges the pair that least increases total within-cluster variance, was used. The dendrogram records the merge history; cutting it at a chosen height gives the clusters. The three long vertical branches in the dendrogram support a three-cluster solution."),
  img("03_dendrogram.png", 1500, 675),
  caption("Figure 3. Truncated Ward dendrogram."),
  p("Four linkage criteria were compared with three clusters each (Table 1). Ward linkage is clearly best. Single and average linkage put 174 of the 178 wines into a single cluster, which gives an Adjusted Rand Index near zero, i.e. no better than random assignment."),
  table(["Linkage", "Silhouette", "Adjusted Rand Index"],
    Object.entries(R.linkage_comparison).map(([k, v]) => [k[0].toUpperCase() + k.slice(1), f(v.silhouette), f(v.ARI)]), [3000, 3013, 3013]),
  caption("Table 1. Hierarchical clustering with different linkage methods (3 clusters)."),

  h2("4.3 Comparing the clusters with the true cultivars"),
  img("04_cluster_pca_plots.png", 2250, 675),
  caption("Figure 4. PCA projection coloured by true cultivar (left), K-Means (centre) and Ward hierarchical clustering (right)."),
  p(`Because the true cultivars are known, the clusters can be validated externally using the Adjusted Rand Index (ARI), which equals 1 for a perfect match and about 0 for random labelling. K-Means achieves an ARI of ${f(R.kmeans_k3_ARI)} and Ward hierarchical clustering ${f(R.hier_ARI)}. Tables 2 and 3 show how the wines of each cultivar are distributed over the clusters (cluster numbers are arbitrary).`),
  table(["True class", "Cluster 0", "Cluster 1", "Cluster 2"], ct(R.crosstab_kmeans), [2726, 2100, 2100, 2100]),
  caption("Table 2. K-Means clusters versus true cultivars."),
  table(["True class", "Cluster 0", "Cluster 1", "Cluster 2"], ct(R.crosstab_hier), [2726, 2100, 2100, 2100]),
  caption("Table 3. Hierarchical (Ward) clusters versus true cultivars."),
  p("K-Means misplaces only 6 of 178 wines (3.4%), all from cultivar 1, which lies between the other two cultivars in feature space. Hierarchical clustering misplaces 13 (7.3%) for the same reason. K-Means is therefore the better clustering on this dataset, although both recover the underlying structure well without ever seeing a label."),

  // ---- 5 ----
  h1("5. Supervised Model Evaluation"),
  h2("5.1 Evaluation metrics used"),
  bullet([run("Cross-validation: ", { bold: true }), run("the training data is split into 5 stratified folds; each fold is held out once while the model trains on the other four. The mean and spread of the scores estimate how well the model generalises, and are less sensitive to one lucky split than a single train/test split.")]),
  bullet([run("Confusion matrix: ", { bold: true }), run("a table of actual versus predicted classes. Off-diagonal cells show exactly which classes are being confused.")]),
  bullet([run("Precision: ", { bold: true }), run("of the wines predicted as a class, the fraction that truly belong to it (TP / (TP + FP)).")]),
  bullet([run("Recall: ", { bold: true }), run("of the wines that truly belong to a class, the fraction found (TP / (TP + FN)).")]),
  bullet([run("F1-score: ", { bold: true }), run("the harmonic mean of precision and recall; macro-averaging gives each class equal weight.")]),
  bullet([run("ROC-AUC: ", { bold: true }), run("the area under the curve of true-positive rate against false-positive rate across all decision thresholds; computed one-vs-rest for the three classes. 1.0 is perfect and 0.5 is random.")]),
  bullet([run("Hyperparameter tuning: ", { bold: true }), run("GridSearchCV exhaustively tries every combination in a parameter grid using cross-validation and keeps the best one.")]),

  h2("5.2 Experimental setup"),
  p("The data was split 75% / 25% into a training set (133 wines) and an untouched test set (45 wines), stratified by cultivar with a fixed random seed. Two classifiers were compared: Logistic Regression (inside a pipeline with a standard scaler) and a Random Forest. All tuning used only the training set."),
  p(`Baseline 5-fold cross-validated accuracy on the training set was ${f(R.cv_baseline["Logistic Regression"].mean)} ± ${f(R.cv_baseline["Logistic Regression"].std)} for Logistic Regression and ${f(R.cv_baseline["Random Forest"].mean)} ± ${f(R.cv_baseline["Random Forest"].std)} for Random Forest.`),

  h2("5.3 Hyperparameter tuning"),
  table(["Model", "Grid searched", "Best parameters", "CV macro-F1"], [
    ["Logistic Regression", "C ∈ {0.01, 0.1, 1, 10, 100}", "C = 0.01", f(R.grid_lr.best_cv_f1_macro)],
    ["Random Forest", "n_estimators ∈ {50, 100, 200}; max_depth ∈ {None, 3, 5, 10}; min_samples_split ∈ {2, 5}", "n_estimators = 50, max_depth = None, min_samples_split = 2", f(R.grid_rf.best_cv_f1_macro)],
  ], [1800, 3226, 2800, 1200]),
  caption("Table 4. GridSearchCV results (5-fold, scoring = macro-F1)."),
  p("For Logistic Regression the best C is the strongest regularisation in the grid. This suggests that, with 13 correlated features and only 133 training samples, a heavily regularised model generalises best. All 24 Random Forest combinations scored between 0.978 and 0.985, so the Random Forest result is not sensitive to these choices."),

  h2("5.4 Test-set results"),
  table(["Metric (macro-averaged)", "Logistic Regression", "Random Forest"], [
    ["Accuracy", f(lr.accuracy), f(rf.accuracy)],
    ["Precision", f(lr.precision_macro), f(rf.precision_macro)],
    ["Recall", f(lr.recall_macro), f(rf.recall_macro)],
    ["F1-score", f(lr.f1_macro), f(rf.f1_macro)],
    ["ROC-AUC (one-vs-rest)", f(lr.roc_auc_ovr), f(rf.roc_auc_ovr)],
  ], [3626, 2700, 2700]),
  caption("Table 5. Hold-out test-set performance of the tuned models (45 wines)."),
  img("05_confusion_matrices.png", 1500, 600),
  caption("Figure 5. Confusion matrices on the test set."),
  p("Both tuned models classify all 45 test wines correctly: every confusion matrix entry lies on the diagonal, and precision, recall, F1 and ROC-AUC are all 1.0 for every class."),
  img("06_roc_curves.png", 900, 750),
  caption("Figure 6. One-vs-rest ROC curves for the tuned Logistic Regression."),

  h2("5.5 A more reliable estimate: cross-validation on the full dataset"),
  p("A perfect score on 45 test samples should be interpreted carefully. With so few samples, one or two errors would change the score noticeably, and the result depends on which wines happened to land in the test set. Repeating the evaluation with 5-fold cross-validation over all 178 wines gives a steadier estimate (Table 6)."),
  table(["Metric (5-fold CV, full data)", "Logistic Regression", "Random Forest"], [
    ["Accuracy", f(cvLR.accuracy), f(cvRF.accuracy)],
    ["Precision (macro)", f(cvLR.precision_macro), f(cvRF.precision_macro)],
    ["Recall (macro)", f(cvLR.recall_macro), f(cvRF.recall_macro)],
    ["F1-score (macro)", f(cvLR.f1_macro), f(cvRF.f1_macro)],
  ], [3626, 2700, 2700]),
  caption("Table 6. Five-fold cross-validated performance on all 178 wines."),
  p("Realistic performance is therefore about 97–98% for both models, with Random Forest marginally ahead. The difference of roughly half a percentage point (about one wine) is too small to declare a winner."),

  h2("5.6 Feature importance"),
  img("07_feature_importance.png", 975, 750),
  caption("Figure 7. Top 10 features by Random Forest importance."),
  p(`The most informative features are ${R.top_features.join(", ")}. These are chemical properties linked to the taste, colour and body of the wine, which is plausible because the cultivars differ in exactly those characteristics.`),

  // ---- 6 ----
  h1("6. Discussion and Limitations"),
  bullet("Unsupervised clustering recovered the three cultivars without labels (ARI 0.90 for K-Means), which shows that the chemical profiles carry strong class structure."),
  bullet("The silhouette score (about 0.28) is moderate, and cultivar 1 overlaps with its neighbours. Internal metrics like silhouette and external metrics like ARI answer different questions and are best used together."),
  bullet("Method choice matters: Ward linkage worked well while single and average linkage failed on the same data."),
  bullet("The Wine dataset is small, clean and well separated, so the near-perfect supervised results will not carry over automatically to noisier real-world data. Only one 75/25 split was used for the hold-out test, so repeated cross-validation was added to give a fairer estimate."),
  bullet("The ROC curves are one-vs-rest. Their area is 1.0 here, so they cannot reveal threshold trade-offs; on harder data they would be more informative."),

  // ---- 7 ----
  h1("7. Conclusion"),
  p("This project applied K-Means and hierarchical clustering with PCA to the Wine dataset and then evaluated two classifiers with cross-validation, a confusion matrix, precision, recall, F1-score, ROC-AUC and hyperparameter tuning. Silhouette analysis and the elbow method both selected k = 3, K-Means outperformed hierarchical clustering (ARI 0.90 versus 0.79), and both tuned classifiers reached about 97–98% cross-validated accuracy and perfect scores on the hold-out test set. Together the results show how clustering can uncover structure in unlabelled data and how a combination of metrics gives a more honest picture of model quality than accuracy alone."),

  // ---- 8 ----
  h1("8. Reproducibility"),
  p("All code is available in the accompanying GitHub repository as a Jupyter notebook (Week3_Clustering_Model_Evaluation.ipynb) and an equivalent script (analysis.py). A fixed random seed (42) is used everywhere, so the numbers in this report can be reproduced exactly."),
];

const doc = new Document({
  creator: "Intern", title: "Week 3 - Unsupervised Learning & Model Evaluation",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: FONT, color: "1F4E79" }, paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: FONT, color: "2E75B6" }, paragraph: { spacing: { before: 260, after: 120 }, outlineLevel: 1, keepNext: true } },
    ],
  },
  numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "Week 3: Unsupervised Learning & Model Evaluation  |  Page ", font: FONT, size: 18, color: "777777" }),
        new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18, color: "777777" })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((b) => { fs.writeFileSync("Week3_Model_Evaluation_Report.docx", b); console.log("saved"); });
