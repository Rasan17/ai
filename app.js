/**
 * NPH ML/Neuralnet Predictor - Main Application Controller
 * Handles user interactions, file ingestion, stepper state, dynamic forms,
 * real-time chart rendering, single & batch inference, and dataset exports.
 * 
 * Conceived, designed and developed by Dr G Narenthiran MB ChB BSc(MedSci)(Hons) FEBNS FRCS(SN)
 */

(function () {
    'use strict';

    // Application State
    const AppState = {
        currentStep: 1,
        theme: 'dark',
        fileName: '',
        parsedData: null,
        columnAnalysis: null,
        columnConfigs: {},
        imputedData: null,
        prepMeta: null,
        trainTestData: null,
        selectedModelId: 'nn',
        trainedModel: null,
        evalResults: null,
        charts: {},
        batchData: null,
        batchPredictions: null
    };

    // DOM Elements
    const DOM = {};

    function initDOM() {
        DOM.themeToggle = document.getElementById('theme-toggle');
        DOM.stepTabs = document.querySelectorAll('.step-tab');
        DOM.stepContents = document.querySelectorAll('.step-content');
        DOM.fileInput = document.getElementById('csv-file-input');
        DOM.dropzone = document.getElementById('upload-dropzone');
        DOM.sampleCardioBtn = document.getElementById('sample-cardio-btn');
        DOM.sampleCohortBtn = document.getElementById('sample-cohort-btn');
        DOM.sampleRegressionBtn = document.getElementById('sample-regression-btn');
        DOM.dataStatsGrid = document.getElementById('data-stats-grid');
        DOM.dataPreviewContainer = document.getElementById('data-preview-container');
        DOM.toStep2Btn = document.getElementById('to-step-2-btn');
        
        DOM.varConfigGrid = document.getElementById('var-config-grid');
        DOM.toStep3Btn = document.getElementById('to-step-3-btn');
        
        DOM.imputeTableBody = document.getElementById('impute-table-body');
        DOM.applyImputationBtn = document.getElementById('apply-imputation-btn');
        DOM.imputeAuditBox = document.getElementById('impute-audit-box');
        DOM.toStep4Btn = document.getElementById('to-step-4-btn');
        
        DOM.modelCards = document.querySelectorAll('.model-card');
        DOM.trainModelBtn = document.getElementById('train-model-btn');
        DOM.trainStatus = document.getElementById('train-status');
        DOM.trainLossChartCanvas = document.getElementById('train-loss-chart');
        DOM.resultsDashboard = document.getElementById('results-dashboard');
        DOM.metricsSummaryRow = document.getElementById('metrics-summary-row');
        DOM.evaluationDetailContainer = document.getElementById('eval-detail-container');
        DOM.retrainStatusBanner = document.getElementById('retrain-status-banner');
        DOM.toStep5Btn = document.getElementById('to-step-5-btn');
        DOM.toStep6Btn = document.getElementById('to-step-6-btn');
        
        DOM.predictFormGrid = document.getElementById('predict-form-grid');
        DOM.runSinglePredictBtn = document.getElementById('run-single-predict-btn');
        DOM.singleResultBanner = document.getElementById('single-result-banner');
        
        DOM.batchFileInput = document.getElementById('batch-file-input');
        DOM.batchDropzone = document.getElementById('batch-dropzone');
        DOM.sampleBatchBtn = document.getElementById('sample-batch-btn');
        DOM.runBatchPredictBtn = document.getElementById('run-batch-predict-btn');
        DOM.batchResultsCard = document.getElementById('batch-results-card');
        DOM.batchTableContainer = document.getElementById('batch-table-container');
        DOM.downloadBatchCsvBtn = document.getElementById('download-batch-csv-btn');
    }

    function setupEventListeners() {
        // Theme toggle
        DOM.themeToggle.addEventListener('click', toggleTheme);

        // Stepper Navigation
        DOM.stepTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const step = parseInt(tab.dataset.step, 10);
                goToStep(step);
            });
        });

        // Next Step Buttons
        DOM.toStep2Btn.addEventListener('click', () => goToStep(2));
        DOM.toStep3Btn.addEventListener('click', () => goToStep(3));
        DOM.toStep4Btn.addEventListener('click', () => goToStep(4));
        DOM.toStep5Btn.addEventListener('click', () => goToStep(5));
        DOM.toStep6Btn.addEventListener('click', () => goToStep(6));

        // CSV Drag and Drop
        DOM.dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            DOM.dropzone.classList.add('drag-active');
        });
        DOM.dropzone.addEventListener('dragleave', () => DOM.dropzone.classList.remove('drag-active'));
        DOM.dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            DOM.dropzone.classList.remove('drag-active');
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                handleFileUpload(e.dataTransfer.files[0]);
            }
        });
        DOM.dropzone.addEventListener('click', () => DOM.fileInput.click());
        DOM.fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                handleFileUpload(e.target.files[0]);
            }
        });

        // Sample Data Loaders
        if (DOM.sampleCardioBtn) {
            DOM.sampleCardioBtn.addEventListener('click', () => loadSampleDataset('universal_medical_cohort.csv', 'Cardiometabolic Risk Cohort (Binary)'));
        }
        DOM.sampleCohortBtn.addEventListener('click', () => loadSampleDataset('medical_shunt_cohort.csv', 'Neurosurgical Shunt Response Cohort'));
        DOM.sampleRegressionBtn.addEventListener('click', () => loadSampleDataset('medical_recovery_score.csv', 'Functional Recovery Score (Continuous Regression)'));

        // Imputation execution
        DOM.applyImputationBtn.addEventListener('click', executeImputation);

        // Model Selection
        DOM.modelCards.forEach(card => {
            card.addEventListener('click', () => {
                DOM.modelCards.forEach(c => c.classList.remove('selected'));
                card.classList.add('selected');
                AppState.selectedModelId = card.dataset.modelId;
                updateHyperparamsVisibility();
            });
        });

        // Train / Test Split interactive synchronization
        const slider = document.getElementById('train-ratio-slider');
        const trainInput = document.getElementById('train-pct-input');
        const testInput = document.getElementById('test-pct-input');
        if (slider) {
            slider.addEventListener('input', (e) => syncTrainTestSplit(parseFloat(e.target.value)));
        }
        if (trainInput) {
            trainInput.addEventListener('input', (e) => {
                const val = parseFloat(e.target.value);
                if (!isNaN(val)) syncTrainTestSplit(val);
            });
        }
        if (testInput) {
            testInput.addEventListener('input', (e) => {
                const val = parseFloat(e.target.value);
                if (!isNaN(val)) syncTrainTestSplit(100 - val);
            });
        }
        document.querySelectorAll('.preset-split-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const trainVal = parseFloat(btn.dataset.train);
                if (!isNaN(trainVal)) syncTrainTestSplit(trainVal);
            });
        });

        // Train Model
        DOM.trainModelBtn.addEventListener('click', startModelTraining);

        // Single Prediction
        DOM.runSinglePredictBtn.addEventListener('click', runSinglePrediction);

        // Batch Prediction
        DOM.batchDropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            DOM.batchDropzone.classList.add('drag-active');
        });
        DOM.batchDropzone.addEventListener('dragleave', () => DOM.batchDropzone.classList.remove('drag-active'));
        DOM.batchDropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            DOM.batchDropzone.classList.remove('drag-active');
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                handleBatchFileUpload(e.dataTransfer.files[0]);
            }
        });
        DOM.batchDropzone.addEventListener('click', () => DOM.batchFileInput.click());
        DOM.batchFileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                handleBatchFileUpload(e.target.files[0]);
            }
        });
        DOM.sampleBatchBtn.addEventListener('click', loadSampleBatchDataset);
        DOM.runBatchPredictBtn.addEventListener('click', executeBatchPrediction);
        DOM.downloadBatchCsvBtn.addEventListener('click', downloadBatchCSV);
    }

    function syncTrainTestSplit(trainVal) {
        let train = Math.min(95, Math.max(50, Math.round(trainVal)));
        let test = 100 - train;

        const slider = document.getElementById('train-ratio-slider');
        const trainInput = document.getElementById('train-pct-input');
        const testInput = document.getElementById('test-pct-input');
        const dispTrain = document.getElementById('split-display-train');
        const dispTest = document.getElementById('split-display-test');
        const infoTrain = document.getElementById('info-train-pct');
        const infoTest = document.getElementById('info-test-pct');

        if (slider) slider.value = train;
        if (trainInput) trainInput.value = train;
        if (testInput) testInput.value = test;
        if (dispTrain) dispTrain.innerText = `${train}% Train`;
        if (dispTest) dispTest.innerText = `${test}% Test`;
        if (infoTrain) infoTrain.innerText = `${train}%`;
        if (infoTest) infoTest.innerText = `${test}%`;
    }

    function toggleTheme() {
        AppState.theme = AppState.theme === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', AppState.theme);
        DOM.themeToggle.innerHTML = AppState.theme === 'dark' ? '☀️ Light' : '🌙 Dark';
        // Redraw active charts to adjust gridline contrast
        if (AppState.charts.loss) AppState.charts.loss.update();
    }

    function goToStep(stepNum) {
        // Enforce prerequisites
        if (stepNum > 1 && !AppState.parsedData) {
            alert("Please upload a CSV file or load a sample dataset first.");
            return;
        }
        if (stepNum > 3 && !AppState.imputedData) {
            alert("Please complete the Data Imputation step first.");
            return;
        }
        if (stepNum > 4 && !AppState.trainedModel) {
            alert("Please train a machine learning model first in Step 4.");
            return;
        }

        AppState.currentStep = stepNum;

        DOM.stepTabs.forEach(tab => {
            const s = parseInt(tab.dataset.step, 10);
            tab.classList.toggle('active', s === stepNum);
            if (s < stepNum) tab.classList.add('completed');
        });

        DOM.stepContents.forEach(content => {
            const s = parseInt(content.dataset.step, 10);
            content.classList.toggle('active', s === stepNum);
        });

        window.scrollTo({ top: 0, behavior: 'smooth' });

        if (stepNum === 2) renderVariableConfigGrid();
        if (stepNum === 3) renderImputationTable();
        if (stepNum === 5) renderSinglePredictionForm();
    }

    function handleFileUpload(file) {
        if (!file.name.endsWith('.csv') && file.type !== 'text/csv') {
            alert("Please upload a valid .csv file.");
            return;
        }
        AppState.fileName = file.name;
        const reader = new FileReader();
        reader.onload = (e) => {
            processCSVContent(e.target.result, file.name);
        };
        reader.readAsText(file);
    }

    async function loadSampleDataset(filename, displayName) {
        try {
            const resp = await fetch(filename);
            if (!resp.ok) throw new Error("Could not load sample dataset file.");
            const text = await resp.text();
            processCSVContent(text, displayName);
        } catch (err) {
            alert(`Error loading sample dataset: ${err.message}`);
        }
    }

    function processCSVContent(csvText, name) {
        try {
            const parsed = DataEngine.parseCSV(csvText);
            AppState.parsedData = parsed;
            AppState.fileName = name;

            const analysis = DataEngine.analyzeColumns(parsed.headers, parsed.rows);
            AppState.columnAnalysis = analysis;

            // Initialize default column configs
            AppState.columnConfigs = {};
            parsed.headers.forEach(col => {
                const a = analysis[col];
                AppState.columnConfigs[col] = {
                    role: a.role,
                    type: a.type,
                    imputationStrategy: a.suggestedImputation,
                    customImputeVal: ''
                };
            });

            renderDataOverview();
            DOM.toStep2Btn.disabled = false;
        } catch (e) {
            alert(`Failed to parse CSV: ${e.message}`);
        }
    }

    function renderDataOverview() {
        const { headers, rows, totalRows, totalCols } = AppState.parsedData;
        const analysis = AppState.columnAnalysis;

        let totalMissing = 0;
        let colsWithMissing = 0;
        headers.forEach(h => {
            totalMissing += analysis[h].missingCount;
            if (analysis[h].missingCount > 0) colsWithMissing++;
        });

        const targetCandidate = headers.find(h => AppState.columnConfigs[h].role === 'target') || headers[headers.length - 1];

        DOM.dataStatsGrid.innerHTML = `
            <div class="stat-card">
                <span class="stat-label">File Name</span>
                <span class="stat-value" style="font-size: 16px; overflow:hidden; text-overflow:ellipsis;">${AppState.fileName}</span>
                <span class="stat-desc">Active dataset</span>
            </div>
            <div class="stat-card">
                <span class="stat-label">Total Observations</span>
                <span class="stat-value">${totalRows}</span>
                <span class="stat-desc">Patient records / rows</span>
            </div>
            <div class="stat-card">
                <span class="stat-label">Total Variables</span>
                <span class="stat-value">${totalCols}</span>
                <span class="stat-desc">Columns detected</span>
            </div>
            <div class="stat-card">
                <span class="stat-label">Missing Cells</span>
                <span class="stat-value" style="color: ${totalMissing > 0 ? 'var(--accent-rose)' : 'var(--accent-emerald)'}">${totalMissing}</span>
                <span class="stat-desc">${colsWithMissing} variable(s) with empty values</span>
            </div>
        `;

        // Render Data Preview Table (First 8 rows)
        const previewRows = rows.slice(0, 8);
        let tableHtml = `
            <table class="data-table">
                <thead>
                    <tr>
                        ${headers.map(h => `<th>${h}</th>`).join('')}
                    </tr>
                </thead>
                <tbody>
                    ${previewRows.map(r => `
                        <tr>
                            ${headers.map(h => {
                                const val = r[h];
                                const isMiss = DataEngine.isMissing(val);
                                return `<td style="${isMiss ? 'color: var(--accent-rose); font-style: italic;' : ''}">${isMiss ? '∅ (Empty)' : val}</td>`;
                            }).join('')}
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
        DOM.dataPreviewContainer.innerHTML = tableHtml;
    }

    // =========================================================================
    // STEP 2: VARIABLE ROLES & TYPES CONFIGURATION
    // =========================================================================
    function renderVariableConfigGrid() {
        const { headers } = AppState.parsedData;
        const analysis = AppState.columnAnalysis;

        let gridHtml = '';
        headers.forEach(col => {
            const a = analysis[col];
            const cfg = AppState.columnConfigs[col];

            gridHtml += `
                <div class="var-card ${cfg.role === 'target' ? 'is-target' : (cfg.role === 'meta' ? 'is-meta' : '')}" id="card-col-${col}">
                    <div class="var-header">
                        <span class="var-name" title="${col}">${col}</span>
                        ${a.missingCount > 0 ? `<span class="pill pill-missing">${a.missingCount} missing (${a.missingPercent}%)</span>` : `<span class="pill" style="background: rgba(16, 185, 129, 0.1); color: #10b981;">Complete</span>`}
                    </div>

                    <div style="font-size: 11.5px; color: var(--text-muted);">
                        ${a.isNumeric ? `Numeric (range: ${a.stats.min} to ${a.stats.max})` : `Distinct values: ${a.distinctCount} (e.g. ${a.distinctValues.slice(0, 3).join(', ')})`}
                    </div>

                    <div class="var-row-control">
                        <label>Variable Role</label>
                        <select class="control-select var-role-select" data-col="${col}">
                            <option value="feature" ${cfg.role === 'feature' ? 'selected' : ''}>Predictor Feature</option>
                            <option value="target" ${cfg.role === 'target' ? 'selected' : ''}>Target (Outcome to Predict)</option>
                            <option value="meta" ${cfg.role === 'meta' ? 'selected' : ''}>Meta / ID (Ignore from Model)</option>
                        </select>
                    </div>

                    <div class="var-row-control">
                        <label>Variable Type</label>
                        <select class="control-select var-type-select" data-col="${col}">
                            <option value="continuous" ${cfg.type === 'continuous' ? 'selected' : ''}>Continuous (Numerical / Scale)</option>
                            <option value="ordinal" ${cfg.type === 'ordinal' ? 'selected' : ''}>Ordinal (Ranked Categorical)</option>
                            <option value="nominal" ${cfg.type === 'nominal' ? 'selected' : ''}>Nominal (Unordered Categorical)</option>
                        </select>
                    </div>
                </div>
            `;
        });

        DOM.varConfigGrid.innerHTML = gridHtml;

        // Attach listeners for role & type updates
        document.querySelectorAll('.var-role-select').forEach(sel => {
            sel.addEventListener('change', (e) => {
                const col = e.target.dataset.col;
                const newRole = e.target.value;

                if (newRole === 'target') {
                    // Reset any previously selected target
                    Object.keys(AppState.columnConfigs).forEach(otherCol => {
                        if (otherCol !== col && AppState.columnConfigs[otherCol].role === 'target') {
                            AppState.columnConfigs[otherCol].role = 'feature';
                            const otherCard = document.getElementById(`card-col-${otherCol}`);
                            if (otherCard) {
                                otherCard.classList.remove('is-target');
                                otherCard.querySelector('.var-role-select').value = 'feature';
                            }
                        }
                    });
                }

                AppState.columnConfigs[col].role = newRole;
                const card = document.getElementById(`card-col-${col}`);
                if (card) {
                    card.classList.toggle('is-target', newRole === 'target');
                    card.classList.toggle('is-meta', newRole === 'meta');
                }
            });
        });

        document.querySelectorAll('.var-type-select').forEach(sel => {
            sel.addEventListener('change', (e) => {
                const col = e.target.dataset.col;
                AppState.columnConfigs[col].type = e.target.value;
                // update default suggested imputation strategy in step 3
                if (e.target.value === 'continuous') {
                    AppState.columnConfigs[col].imputationStrategy = 'median';
                } else {
                    AppState.columnConfigs[col].imputationStrategy = 'mode';
                }
            });
        });
    }

    // =========================================================================
    // STEP 3: MISSING DATA IMPUTATION
    // =========================================================================
    function renderImputationTable() {
        const { headers } = AppState.parsedData;
        const analysis = AppState.columnAnalysis;

        let rowsHtml = '';
        let missingColCount = 0;

        headers.forEach(col => {
            const a = analysis[col];
            const cfg = AppState.columnConfigs[col];

            // Only show columns with missing values and non-meta roles
            if (cfg.role === 'meta') return;
            if (a.missingCount === 0) return;

            missingColCount++;

            let strategyOptionsHtml = '';
            if (cfg.type === 'continuous') {
                strategyOptionsHtml = `
                    <option value="median" ${cfg.imputationStrategy === 'median' ? 'selected' : ''}>Median (${a.stats.median})</option>
                    <option value="mean" ${cfg.imputationStrategy === 'mean' ? 'selected' : ''}>Mean (${a.stats.mean})</option>
                    <option value="knn" ${cfg.imputationStrategy === 'knn' ? 'selected' : ''}>KNN (K-Nearest Neighbors)</option>
                    <option value="mode" ${cfg.imputationStrategy === 'mode' ? 'selected' : ''}>Mode (${a.stats.mode})</option>
                    <option value="constant" ${cfg.imputationStrategy === 'constant' ? 'selected' : ''}>Constant Number...</option>
                    <option value="drop" ${cfg.imputationStrategy === 'drop' ? 'selected' : ''}>Drop Rows with Missing</option>
                `;
            } else if (cfg.type === 'ordinal') {
                strategyOptionsHtml = `
                    <option value="mode" ${cfg.imputationStrategy === 'mode' ? 'selected' : ''}>Mode / Most Frequent (${a.stats.mode})</option>
                    <option value="median" ${cfg.imputationStrategy === 'median' ? 'selected' : ''}>Median Rank (${a.stats.median})</option>
                    <option value="constant" ${cfg.imputationStrategy === 'constant' ? 'selected' : ''}>Constant Rank...</option>
                    <option value="drop" ${cfg.imputationStrategy === 'drop' ? 'selected' : ''}>Drop Rows with Missing</option>
                `;
            } else {
                strategyOptionsHtml = `
                    <option value="mode" ${cfg.imputationStrategy === 'mode' ? 'selected' : ''}>Mode / Most Frequent ("${a.stats.mode}")</option>
                    <option value="constant" ${cfg.imputationStrategy === 'constant' ? 'selected' : ''}>New Category ("Missing" / "Unknown")</option>
                    <option value="drop" ${cfg.imputationStrategy === 'drop' ? 'selected' : ''}>Drop Rows with Missing</option>
                `;
            }

            rowsHtml += `
                <tr>
                    <td style="font-weight: 600;">${col}</td>
                    <td><span class="pill pill-${cfg.role}">${cfg.role}</span></td>
                    <td><span class="pill pill-${cfg.type}">${cfg.type}</span></td>
                    <td><span class="pill pill-missing">${a.missingCount} / ${a.total} (${a.missingPercent}%)</span></td>
                    <td>
                        <select class="control-select impute-strat-select" data-col="${col}" style="min-width: 200px;">
                            ${strategyOptionsHtml}
                        </select>
                    </td>
                    <td>
                        <input type="text" class="control-input impute-custom-val" data-col="${col}" placeholder="${cfg.type === 'continuous' ? '0' : 'Missing'}" value="${cfg.customImputeVal || ''}" style="${cfg.imputationStrategy === 'constant' ? 'display:inline-block; width: 110px;' : 'display:none; width: 110px;'}">
                    </td>
                </tr>
            `;
        });

        if (missingColCount === 0) {
            DOM.imputeTableBody.innerHTML = `
                <tr>
                    <td colspan="6" style="text-align: center; padding: 28px; color: var(--accent-emerald);">
                        ✓ No missing data detected in active variables. The dataset is complete and ready for machine learning!
                    </td>
                </tr>
            `;
        } else {
            DOM.imputeTableBody.innerHTML = rowsHtml;
        }

        // Attach listeners
        document.querySelectorAll('.impute-strat-select').forEach(sel => {
            sel.addEventListener('change', (e) => {
                const col = e.target.dataset.col;
                const strat = e.target.value;
                AppState.columnConfigs[col].imputationStrategy = strat;
                const customInput = document.querySelector(`.impute-custom-val[data-col="${col}"]`);
                if (customInput) {
                    customInput.style.display = strat === 'constant' ? 'inline-block' : 'none';
                }
            });
        });

        document.querySelectorAll('.impute-custom-val').forEach(input => {
            input.addEventListener('input', (e) => {
                const col = e.target.dataset.col;
                AppState.columnConfigs[col].customImputeVal = e.target.value;
            });
        });
    }

    function executeImputation() {
        try {
            const { rows } = AppState.parsedData;
            const res = DataEngine.imputeDataset(rows, AppState.columnConfigs, AppState.columnAnalysis);
            AppState.imputedData = res;

            let auditHtml = `
                <div style="background: var(--bg-surface-elevated); border: 1px solid var(--accent-emerald); border-radius: var(--radius-md); padding: 16px;">
                    <h4 style="color: var(--accent-emerald); display: flex; align-items: center; gap: 8px;">
                        ✓ Imputation Complete: ${res.totalRowsAfter} clean records ready
                    </h4>
                    <ul style="margin: 10px 0 0 20px; font-size: 13px; color: var(--text-secondary);">
                        ${res.audit.length === 0 ? '<li>No missing cells required imputation.</li>' : res.audit.map(a => `
                            <li><strong>${a.column || 'Rows'}</strong>: ${a.strategy || a.type} &rarr; ${a.imputedCount || ''} values imputed (${a.filledValue || a.description || ''})</li>
                        `).join('')}
                    </ul>
                </div>
            `;
            DOM.imputeAuditBox.innerHTML = auditHtml;
            DOM.toStep4Btn.disabled = false;
        } catch (e) {
            alert(`Imputation error: ${e.message}`);
        }
    }

    // =========================================================================
    // STEP 4: MODEL TRAINING & EVALUATION
    // =========================================================================
    function updateHyperparamsVisibility() {
        const isNN = AppState.selectedModelId === 'nn';
        const isRF = AppState.selectedModelId === 'rf';
        const isGLM = AppState.selectedModelId === 'glm';
        const isKNN = AppState.selectedModelId === 'knn';
        const isEnsemble = AppState.selectedModelId === 'ensemble';

        document.getElementById('nn-params-box').style.display = isNN ? 'grid' : 'none';
        document.getElementById('rf-params-box').style.display = isRF ? 'grid' : 'none';
        document.getElementById('glm-params-box').style.display = isGLM ? 'grid' : 'none';
        document.getElementById('knn-params-box').style.display = isKNN ? 'grid' : 'none';
        document.getElementById('ensemble-params-box').style.display = isEnsemble ? 'grid' : 'none';
    }

    async function startModelTraining() {
        if (!AppState.imputedData) {
            alert("Please execute Data Imputation first.");
            return;
        }

        try {
            DOM.trainModelBtn.disabled = true;
            DOM.trainStatus.innerHTML = `<span style="color: var(--accent-cyan); display: inline-flex; align-items: center; gap: 6px;">⚙️ Preparing data partitions and hyperparameter configurations...</span>`;

            // Prepare features and target
            const prep = DataEngine.prepareFeaturesAndTarget(AppState.imputedData.imputedRows, AppState.columnConfigs);
            AppState.prepMeta = prep;

            // Train / Test split percentage (default 80% train / 20% test)
            const trainPctInput = document.getElementById('train-pct-input');
            const trainPct = trainPctInput ? parseFloat(trainPctInput.value) : 80;
            const trainRatio = Math.min(0.95, Math.max(0.50, (trainPct || 80) / 100));
            const testPct = Math.round((1 - trainRatio) * 100);

            const split = DataEngine.trainTestSplit(prep.X, prep.y, trainRatio, prep.isClassification);
            AppState.trainTestData = split;

            // Hyperparameters
            const hyperparams = {};
            if (AppState.selectedModelId === 'nn') {
                hyperparams.epochs = parseInt(document.getElementById('nn-epochs-input').value, 10) || 50;
                hyperparams.learningRate = parseFloat(document.getElementById('nn-lr-input').value) || 0.005;
                hyperparams.hiddenLayers = [
                    parseInt(document.getElementById('nn-h1-input').value, 10) || 64,
                    parseInt(document.getElementById('nn-h2-input').value, 10) || 32
                ];
                hyperparams.dropoutRate = parseFloat(document.getElementById('nn-dropout-input').value) || 0.1;
                hyperparams.activation = document.getElementById('nn-act-select').value || 'relu';
            } else if (AppState.selectedModelId === 'rf') {
                hyperparams.numTrees = parseInt(document.getElementById('rf-trees-input').value, 10) || 30;
                hyperparams.maxDepth = parseInt(document.getElementById('rf-depth-input').value, 10) || 6;
            } else if (AppState.selectedModelId === 'glm') {
                hyperparams.lr = parseFloat(document.getElementById('glm-lr-input').value) || 0.05;
                hyperparams.epochs = parseInt(document.getElementById('glm-epochs-input').value, 10) || 150;
            } else if (AppState.selectedModelId === 'knn') {
                hyperparams.k = parseInt(document.getElementById('knn-k-input').value, 10) || 5;
            } else if (AppState.selectedModelId === 'ensemble') {
                const selectedBases = [];
                if (document.getElementById('ens-include-nn').checked) selectedBases.push('nn');
                if (document.getElementById('ens-include-rf').checked) selectedBases.push('rf');
                if (document.getElementById('ens-include-glm').checked) selectedBases.push('glm');
                if (document.getElementById('ens-include-knn').checked) selectedBases.push('knn');
                if (document.getElementById('ens-include-nb').checked && prep.isClassification) selectedBases.push('nb');
                if (selectedBases.length === 0) selectedBases.push('rf', 'glm');

                hyperparams.voting = document.getElementById('ensemble-voting-select').value || 'soft';
                hyperparams.baseModels = selectedBases;
                hyperparams.nnEpochs = 35;
                hyperparams.rfTrees = 25;
                hyperparams.rfDepth = 6;
                hyperparams.knnK = 5;
            }

            // -------------------------------------------------------------
            // PHASE 1: SPLIT TRAINING & VALIDATION ACCURACY BENCHMARK
            // -------------------------------------------------------------
            DOM.trainStatus.innerHTML = `<span style="color: var(--accent-cyan);">🚀 Phase 1/2: Training on ${Math.round(trainRatio * 100)}% partition (${split.trainX.length} cases) to evaluate ${testPct}% test accuracy...</span>`;

            // Chart setup for NN / Ensemble training progress
            setupTrainingLossChart();

            const valModel = ModelEngine.createModel(
                AppState.selectedModelId,
                prep.isClassification,
                prep.targetClasses.length || 2,
                hyperparams
            );

            // Fit validation model on train split
            if (AppState.selectedModelId === 'nn') {
                await valModel.fit(split.trainX, split.trainY, split.testX, split.testY, (epoch, total, logs) => {
                    DOM.trainStatus.innerHTML = `<span style="color: var(--accent-cyan);">Phase 1 (${Math.round(trainRatio * 100)}% Train): Epoch ${epoch}/${total} &mdash; Train Loss: ${logs.loss.toFixed(4)} ${logs.val_loss !== undefined ? `| Test Loss: ${logs.val_loss.toFixed(4)}` : ''}</span>`;
                    updateLossChart(epoch, logs.loss, logs.val_loss);
                });
            } else if (AppState.selectedModelId === 'ensemble') {
                await valModel.fit(split.trainX, split.trainY, split.testX, split.testY, (curr, total, info) => {
                    DOM.trainStatus.innerHTML = `<span style="color: var(--accent-cyan);">${info.stage || `Phase 1 (${Math.round(trainRatio * 100)}% Train): Base ${curr}/${total}...`}</span>`;
                    if (info.loss !== undefined) {
                        updateLossChart(curr, info.loss, info.val_loss);
                    }
                });
            } else {
                valModel.fit(split.trainX, split.trainY);
            }

            // Benchmark out-of-sample accuracy on held-out test split
            const evalRes = ModelEngine.evaluate(valModel, split.testX, split.testY, prep.isClassification, prep.targetClasses);
            evalRes.trainRatio = trainRatio;
            evalRes.trainPct = Math.round(trainRatio * 100);
            evalRes.testPct = testPct;
            evalRes.trainCount = split.trainX.length;
            evalRes.testCount = split.testX.length;
            evalRes.totalCount = prep.X.length;
            AppState.evalResults = evalRes;

            const accuracySummary = evalRes.isClassification
                ? `Test Accuracy: ${(evalRes.accuracy * 100).toFixed(1)}% (ROC-AUC: ${evalRes.rocAuc !== null ? evalRes.rocAuc.toFixed(3) : 'N/A'})`
                : `Test R²: ${evalRes.r2.toFixed(3)} (RMSE: ${evalRes.rmse.toFixed(3)})`;

            // -------------------------------------------------------------
            // PHASE 2: RETRAIN FINAL MODEL WITH 100% OF DATA
            // -------------------------------------------------------------
            DOM.trainStatus.innerHTML = `<span style="color: var(--accent-amber); font-weight: 600;">✓ Phase 1 Complete [${accuracySummary}]. Now Phase 2: Retraining final model with 100% of data (N=${prep.X.length} patients)...</span>`;

            // Yield briefly to UI thread
            await new Promise(r => setTimeout(r, 60));

            const fullModel = ModelEngine.createModel(
                AppState.selectedModelId,
                prep.isClassification,
                prep.targetClasses.length || 2,
                hyperparams
            );

            if (AppState.selectedModelId === 'nn') {
                setupTrainingLossChart();
                await fullModel.fit(prep.X, prep.y, null, null, (epoch, total, logs) => {
                    DOM.trainStatus.innerHTML = `<span style="color: var(--accent-cyan);">Phase 2 (100% Retraining): Epoch ${epoch}/${total} &mdash; Full Cohort Loss: ${logs.loss.toFixed(4)}</span>`;
                    updateLossChart(epoch, logs.loss);
                });
            } else if (AppState.selectedModelId === 'ensemble') {
                await fullModel.fit(prep.X, prep.y, null, null, (curr, total, info) => {
                    DOM.trainStatus.innerHTML = `<span style="color: var(--accent-cyan);">${info.stage || `Phase 2 (100% Retraining): Base Model ${curr}/${total}...`}</span>`;
                });
            } else {
                fullModel.fit(prep.X, prep.y);
            }

            // Assign fully retrained model for real-time single and batch prediction
            AppState.trainedModel = fullModel;

            DOM.trainStatus.innerHTML = `
                <div style="display: flex; flex-direction: column; gap: 3px;">
                    <span style="color: var(--accent-emerald); font-weight: 700;">
                        ✓ Accuracy Evaluated on ${testPct}% Test Set &amp; Model Fully Retrained on 100% of Cohort Data!
                    </span>
                    <span style="font-size: 11.5px; color: var(--text-secondary);">
                        Reported Benchmark: <strong>${accuracySummary}</strong> &bull; Active Deployed Model: <strong>100% Cohort (N = ${prep.X.length} patients)</strong>
                    </span>
                </div>
            `;
            DOM.trainModelBtn.disabled = false;
            DOM.toStep5Btn.disabled = false;
            DOM.toStep6Btn.disabled = false;

            renderEvaluationDashboard();
        } catch (e) {
            DOM.trainStatus.innerHTML = `<span style="color: var(--accent-rose);">Training failed: ${e.message}</span>`;
            DOM.trainModelBtn.disabled = false;
            alert(`Training error: ${e.message}`);
        }
    }

    function setupTrainingLossChart() {
        const ctx = DOM.trainLossChartCanvas.getContext('2d');
        if (AppState.charts.loss) {
            AppState.charts.loss.destroy();
        }

        const isDark = AppState.theme === 'dark';
        AppState.charts.loss = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Training Loss',
                        data: [],
                        borderColor: '#06b6d4',
                        backgroundColor: 'rgba(6, 182, 212, 0.1)',
                        fill: true,
                        tension: 0.3,
                        pointRadius: 2
                    },
                    {
                        label: 'Validation Loss',
                        data: [],
                        borderColor: '#a855f7',
                        backgroundColor: 'rgba(168, 85, 247, 0.1)',
                        fill: true,
                        tension: 0.3,
                        pointRadius: 2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: { color: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)' },
                        ticks: { color: isDark ? '#94a3b8' : '#64748b' },
                        title: { display: true, text: 'Epoch', color: isDark ? '#94a3b8' : '#64748b' }
                    },
                    y: {
                        grid: { color: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)' },
                        ticks: { color: isDark ? '#94a3b8' : '#64748b' },
                        title: { display: true, text: 'Loss', color: isDark ? '#94a3b8' : '#64748b' }
                    }
                },
                plugins: {
                    legend: { labels: { color: isDark ? '#f8fafc' : '#0f172a' } }
                }
            }
        });
    }

    function updateLossChart(epoch, trainLoss, valLoss) {
        if (!AppState.charts.loss) return;
        const chart = AppState.charts.loss;
        chart.data.labels.push(epoch);
        chart.data.datasets[0].data.push(trainLoss);
        if (valLoss !== undefined) {
            chart.data.datasets[1].data.push(valLoss);
        }
        chart.update('none');
    }

    function renderEvaluationDashboard() {
        const evalRes = AppState.evalResults;
        const prep = AppState.prepMeta;
        DOM.resultsDashboard.style.display = 'flex';

        const trainPct = evalRes.trainPct || 80;
        const testPct = evalRes.testPct || 20;
        const trainCount = evalRes.trainCount || Math.round(evalRes.totalCount * 0.8);
        const testCount = evalRes.testCount || (evalRes.totalCount - trainCount);
        const totalCount = evalRes.totalCount || evalRes.totalSamples;

        // Two-Phase Retraining Notice Banner
        if (DOM.retrainStatusBanner) {
            DOM.retrainStatusBanner.innerHTML = `
                <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.28); border-radius: var(--radius-md); padding: 12px 16px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <div style="font-weight: 700; font-size: 13.5px; color: var(--accent-emerald); display: flex; align-items: center; gap: 6px;">
                            <span>✓</span> Validation Evaluated on Held-Out ${testPct}% Test Set &bull; Deployed Model Retrained on 100% of Data
                        </div>
                        <div style="font-size: 12px; color: var(--text-secondary); margin-top: 3px;">
                            Metrics below reflect out-of-sample accuracy on the held-out <strong>${testPct}% test split (${testCount} cases)</strong> after training on <strong>${trainPct}% (${trainCount} cases)</strong>. The active deployed model was then retrained on <strong>100% of all data (${totalCount} cases)</strong> for maximum diagnostic accuracy.
                        </div>
                    </div>
                    <span class="pill pill-target" style="background: rgba(16, 185, 129, 0.18); color: var(--accent-emerald); border-color: rgba(16, 185, 129, 0.4); font-size: 12px; font-weight: 700;">
                        100% Retrained Model Active
                    </span>
                </div>
            `;
        }

        if (evalRes.isClassification) {
            DOM.metricsSummaryRow.innerHTML = `
                <div class="metric-pill-card">
                    <span class="m-title">Test Accuracy (${testPct}% Split)</span>
                    <span class="m-val">${(evalRes.accuracy * 100).toFixed(1)}%</span>
                </div>
                <div class="metric-pill-card">
                    <span class="m-title">Macro F1-Score</span>
                    <span class="m-val">${(evalRes.macroF1 * 100).toFixed(1)}%</span>
                </div>
                <div class="metric-pill-card">
                    <span class="m-title">Macro Precision</span>
                    <span class="m-val">${(evalRes.macroPrecision * 100).toFixed(1)}%</span>
                </div>
                <div class="metric-pill-card">
                    <span class="m-title">Macro Recall</span>
                    <span class="m-val">${(evalRes.macroRecall * 100).toFixed(1)}%</span>
                </div>
                ${evalRes.rocAuc !== null ? `
                    <div class="metric-pill-card">
                        <span class="m-title">ROC AUC</span>
                        <span class="m-val">${(evalRes.rocAuc).toFixed(3)}</span>
                    </div>
                ` : ''}
            `;

            // Confusion Matrix Table
            const cm = evalRes.confMatrix;
            const classes = prep.targetClasses;
            let cmHtml = `
                <div style="display: flex; flex-direction: column; gap: 16px;">
                    <h3 style="font-size: 15px; color: var(--text-primary);">Confusion Matrix (Held-out ${testPct}% Test Cohort: ${testCount} Patients)</h3>
                    <div class="table-container" style="max-width: 500px;">
                        <table class="cm-table">
                            <thead>
                                <tr>
                                    <th>Actual \\ Predicted</th>
                                    ${classes.map(c => `<th>${c}</th>`).join('')}
                                </tr>
                            </thead>
                            <tbody>
                                ${classes.map((actualClass, r) => `
                                    <tr>
                                        <th style="text-align: left;">${actualClass}</th>
                                        ${classes.map((_, c) => {
                                            const val = cm[r][c];
                                            const isHit = r === c;
                                            return `<td class="${isHit ? 'cm-cell-hit' : 'cm-cell-miss'}">${val}</td>`;
                                        }).join('')}
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            `;
            DOM.evaluationDetailContainer.innerHTML = cmHtml;
        } else {
            // Regression Metrics
            DOM.metricsSummaryRow.innerHTML = `
                <div class="metric-pill-card">
                    <span class="m-title">Test R² Score (${testPct}% Split)</span>
                    <span class="m-val">${evalRes.r2.toFixed(3)}</span>
                </div>
                <div class="metric-pill-card">
                    <span class="m-title">RMSE</span>
                    <span class="m-val">${evalRes.rmse.toFixed(3)}</span>
                </div>
                <div class="metric-pill-card">
                    <span class="m-title">MAE</span>
                    <span class="m-val">${evalRes.mae.toFixed(3)}</span>
                </div>
                <div class="metric-pill-card">
                    <span class="m-title">MSE</span>
                    <span class="m-val">${evalRes.mse.toFixed(3)}</span>
                </div>
            `;
            DOM.evaluationDetailContainer.innerHTML = `
                <p style="font-size: 13px; color: var(--text-secondary);">
                    Model validated on held-out ${testPct}% test split (${testCount} patients) with R² = ${evalRes.r2} and Mean Absolute Error = ${evalRes.mae}. The active model was then retrained on 100% of data (${totalCount} patients).
                </p>
            `;
        }
    }

    // =========================================================================
    // STEP 5: SINGLE PREDICTION (CLINICAL PATIENT PREDICTOR)
    // =========================================================================
    function renderSinglePredictionForm() {
        if (!AppState.prepMeta) return;

        const { featureCols, featureMetadata } = AppState.prepMeta;
        let formHtml = '';

        featureCols.forEach(col => {
            const meta = featureMetadata[col];
            const cfg = AppState.columnConfigs[col];

            if (meta.type === 'continuous') {
                const step = (meta.max - meta.min) > 20 ? '1' : '0.01';
                formHtml += `
                    <div class="form-field">
                        <label>
                            <span>${col}</span>
                            <span class="field-hint">Range: ${meta.min} &ndash; ${meta.max}</span>
                        </label>
                        <input type="number" step="${step}" class="control-input single-input-field" data-col="${col}" value="${meta.mean.toFixed(2)}">
                    </div>
                `;
            } else if (meta.type === 'ordinal') {
                formHtml += `
                    <div class="form-field">
                        <label>
                            <span>${col}</span>
                            <span class="field-hint">Ordinal Rank</span>
                        </label>
                        <select class="control-select single-input-field" data-col="${col}">
                            ${meta.ranks.map(rk => `<option value="${rk}">${rk}</option>`).join('')}
                        </select>
                    </div>
                `;
            } else {
                // Nominal
                formHtml += `
                    <div class="form-field">
                        <label>
                            <span>${col}</span>
                            <span class="field-hint">Categorical</span>
                        </label>
                        <select class="control-select single-input-field" data-col="${col}">
                            ${meta.categories.map(cat => `<option value="${cat}">${cat}</option>`).join('')}
                        </select>
                    </div>
                `;
            }
        });

        DOM.predictFormGrid.innerHTML = formHtml;
    }

    function runSinglePrediction() {
        if (!AppState.trainedModel || !AppState.prepMeta) {
            alert("Model is not trained yet.");
            return;
        }

        const inputRow = {};
        document.querySelectorAll('.single-input-field').forEach(field => {
            const col = field.dataset.col;
            inputRow[col] = field.value;
        });

        const xVector = DataEngine.transformSingleRow(inputRow, AppState.prepMeta);
        const result = AppState.trainedModel.predictSingle(xVector);
        const prep = AppState.prepMeta;

        DOM.singleResultBanner.style.display = 'flex';

        let ensembleDetailsHtml = '';
        if (result.ensembleDetails && result.ensembleDetails.length > 0) {
            const weightsStr = result.ensembleDetails.map(d => `${d.id.toUpperCase()}: ${Math.round((d.normalizedWeight !== undefined ? d.normalizedWeight : d.weight) * 100)}%`).join(' &bull; ');
            ensembleDetailsHtml = `
                <div style="font-size: 11px; color: var(--accent-cyan); margin-top: 5px; font-weight: 600;">
                    Ensemble Blend: ${weightsStr}
                </div>
            `;
        }

        if (prep.isClassification) {
            const predictedLabel = prep.targetIdxToClass[result.pred] || result.pred;
            const confidencePercent = result.confidence ? (result.confidence * 100).toFixed(1) : '90.0';

            let probBarsHtml = '';
            if (result.probabilities) {
                probBarsHtml = Object.keys(result.probabilities).map(clsIdx => {
                    const clsLabel = prep.targetIdxToClass[clsIdx] || clsIdx;
                    const p = (result.probabilities[clsIdx] * 100).toFixed(1);
                    return `
                        <div style="font-size: 12px; display: flex; justify-content: space-between; gap: 10px;">
                            <span>${clsLabel}:</span>
                            <strong>${p}%</strong>
                        </div>
                    `;
                }).join('');
            }

            DOM.singleResultBanner.innerHTML = `
                <div class="banner-left">
                    <h3>Predicted ${prep.targetCol}</h3>
                    <div class="pred-outcome-val">${predictedLabel}</div>
                    <div style="font-size: 13px; color: var(--text-secondary); margin-top: 6px;">
                        Model: <strong>${AppState.selectedModelId.toUpperCase()}</strong> &bull; Target: <em>${prep.targetCol}</em>
                    </div>
                    ${ensembleDetailsHtml}
                </div>
                <div class="banner-confidence">
                    <span style="font-size: 12px; color: var(--text-muted); text-transform: uppercase;">Confidence</span>
                    <span style="font-size: 26px; font-weight: 800; color: var(--accent-emerald);">${confidencePercent}%</span>
                    <div class="prob-meter-bar">
                        <div class="prob-meter-fill" style="width: ${confidencePercent}%;"></div>
                    </div>
                    <div style="margin-top: 8px;">
                        ${probBarsHtml}
                    </div>
                </div>
            `;
        } else {
            DOM.singleResultBanner.innerHTML = `
                <div class="banner-left">
                    <h3>Predicted Continuous ${prep.targetCol}</h3>
                    <div class="pred-outcome-val">${result.pred}</div>
                    <div style="font-size: 13px; color: var(--text-secondary); margin-top: 6px;">
                        Model: <strong>${AppState.selectedModelId.toUpperCase()}</strong> &bull; Unit/Scale: <em>${prep.targetCol}</em>
                    </div>
                    ${ensembleDetailsHtml}
                </div>
            `;
        }
    }

    // =========================================================================
    // STEP 6: BATCH PREDICTION FACILITY
    // =========================================================================
    function handleBatchFileUpload(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            processBatchCSV(e.target.result, file.name);
        };
        reader.readAsText(file);
    }

    async function loadSampleBatchDataset() {
        try {
            const resp = await fetch('medical_batch_unseen_cases.csv');
            if (!resp.ok) throw new Error("Could not load sample batch file.");
            const text = await resp.text();
            processBatchCSV(text, 'Sample Unseen Medical Cohort');
        } catch (e) {
            alert(`Error: ${e.message}`);
        }
    }

    function processBatchCSV(csvText, name) {
        try {
            const parsed = DataEngine.parseCSV(csvText);
            AppState.batchData = parsed;
            DOM.runBatchPredictBtn.disabled = false;
            alert(`Uploaded batch file "${name}" with ${parsed.totalRows} records.`);
        } catch (e) {
            alert(`Batch CSV parse error: ${e.message}`);
        }
    }

    function executeBatchPrediction() {
        if (!AppState.trainedModel || !AppState.prepMeta) {
            alert("Please train a model in Step 4 first.");
            return;
        }
        if (!AppState.batchData) {
            alert("Please upload a batch CSV file first.");
            return;
        }

        const { headers, rows } = AppState.batchData;
        const prep = AppState.prepMeta;
        const results = [];

        rows.forEach(r => {
            const xVector = DataEngine.transformSingleRow(r, prep);
            const predRes = AppState.trainedModel.predictSingle(xVector);

            let predictedLabel = predRes.pred;
            let confidence = 'N/A';
            if (prep.isClassification) {
                predictedLabel = prep.targetIdxToClass[predRes.pred] || predRes.pred;
                confidence = predRes.confidence ? `${(predRes.confidence * 100).toFixed(1)}%` : '90.0%';
            }

            const rowCopy = { ...r };
            rowCopy[`Predicted_${prep.targetCol}`] = predictedLabel;
            rowCopy[`Prediction_Confidence`] = confidence;
            results.push(rowCopy);
        });

        AppState.batchPredictions = results;
        DOM.batchResultsCard.style.display = 'flex';
        DOM.downloadBatchCsvBtn.disabled = false;

        // Render preview table
        const resultHeaders = Object.keys(results[0]);
        const previewSlice = results.slice(0, 10);

        let tableHtml = `
            <table class="data-table">
                <thead>
                    <tr>
                        ${resultHeaders.map(h => `<th style="${h.startsWith('Predicted_') ? 'color: var(--accent-cyan); background: rgba(6,182,212,0.1);' : ''}">${h}</th>`).join('')}
                    </tr>
                </thead>
                <tbody>
                    ${previewSlice.map(r => `
                        <tr>
                            ${resultHeaders.map(h => `
                                <td style="${h.startsWith('Predicted_') ? 'font-weight: 700; color: var(--accent-cyan); background: rgba(6,182,212,0.05);' : ''}">${r[h]}</td>
                            `).join('')}
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;

        DOM.batchTableContainer.innerHTML = tableHtml;
    }

    function downloadBatchCSV() {
        if (!AppState.batchPredictions || AppState.batchPredictions.length === 0) return;

        const headers = Object.keys(AppState.batchPredictions[0]);
        const lines = [headers.join(',')];

        AppState.batchPredictions.forEach(row => {
            const line = headers.map(h => {
                let val = String(row[h] !== undefined ? row[h] : '');
                if (val.includes(',') || val.includes('"')) {
                    val = `"${val.replace(/"/g, '""')}"`;
                }
                return val;
            }).join(',');
            lines.push(line);
        });

        const csvString = lines.join('\r\n');
        const blob = new Blob([csvString], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `Universal_Medical_Predictions_${new Date().toISOString().slice(0, 10)}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    // Initialize application when DOM is ready
    document.addEventListener('DOMContentLoaded', () => {
        initDOM();
        setupEventListeners();
        updateHyperparamsVisibility();
    });

})();
