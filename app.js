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
        batchPredictions: null,
        // Validation Protocol (Option A vs Option B)
        validationMode: 'split', // 'split' (Option A) or 'separate' (Option B)
        separateTestRawRows: null,
        separateTestFileName: '',
        separateTestData: null,
        // Model Testing Scope (One Model, Multiple Models, All Models)
        modelScope: 'single', // 'single', 'multi', 'all'
        selectedModelIds: ['nn'],
        activeHyperparamModelId: 'nn',
        allTestedResults: null,
        deployedModelId: 'nn'
    };

    // Model Metadata & Lookup
    const MODEL_INFO = {
        nn: { name: 'Deep Neural Network (MLP)', short: 'Neural Network', tag: 'Deep Learning', pillClass: 'pill-target', icon: '🧠' },
        rf: { name: 'Random Forest', short: 'Random Forest', tag: 'Ensemble Trees', pillClass: 'pill-feature', icon: '🌲' },
        glm: { name: 'Logistic / Ridge Regression', short: 'GLM / Regression', tag: 'Generalized Linear', pillClass: 'pill-continuous', icon: '📈' },
        knn: { name: 'k-Nearest Neighbors (k-NN)', short: 'k-NN', tag: 'Instance-Based', pillClass: 'pill-ordinal', icon: '🎯' },
        nb: { name: 'Gaussian Naive Bayes', short: 'Naive Bayes', tag: 'Probabilistic', pillClass: 'pill-nominal', icon: '🎲' },
        ensemble: { name: 'Ensemble Super Learner', short: 'Ensemble Meta-Learner', tag: 'Meta-Ensemble', pillClass: 'pill-target', icon: '⚡' }
    };
    const ALL_MODEL_IDS = ['nn', 'rf', 'glm', 'knn', 'nb', 'ensemble'];

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
        
        // Model Selection & Scope Elements
        DOM.modelCards = document.querySelectorAll('.model-card');
        DOM.modelSelectCheckboxes = document.querySelectorAll('.model-select-checkbox');
        DOM.scopeSingle = document.getElementById('scope-single');
        DOM.scopeMulti = document.getElementById('scope-multi');
        DOM.scopeAll = document.getElementById('scope-all');
        DOM.scopeSingleLabel = document.getElementById('scope-single-label');
        DOM.scopeMultiLabel = document.getElementById('scope-multi-label');
        DOM.scopeAllLabel = document.getElementById('scope-all-label');
        DOM.btnSelectAllModels = document.getElementById('btn-select-all-models');
        DOM.btnSelectTopModels = document.getElementById('btn-select-top-models');
        DOM.btnClearModelSelection = document.getElementById('btn-clear-model-selection');
        DOM.activeHyperparamSelect = document.getElementById('active-hyperparam-select');

        DOM.trainModelBtn = document.getElementById('train-model-btn');
        DOM.trainStatus = document.getElementById('train-status');
        DOM.trainLossChartCanvas = document.getElementById('train-loss-chart');
        DOM.resultsDashboard = document.getElementById('results-dashboard');
        DOM.metricsSummaryRow = document.getElementById('metrics-summary-row');
        DOM.evaluationDetailContainer = document.getElementById('eval-detail-container');
        DOM.retrainStatusBanner = document.getElementById('retrain-status-banner');
        DOM.multiModelLeaderboardContainer = document.getElementById('multi-model-leaderboard-container');
        DOM.multiModelCmContainer = document.getElementById('multi-model-cm-container');
        DOM.dashboardSingleHeading = document.getElementById('dashboard-single-heading');

        DOM.toStep5Btn = document.getElementById('to-step-5-btn');
        DOM.toStep6Btn = document.getElementById('to-step-6-btn');
        
        // Option A vs Option B Protocol Elements
        DOM.valModeA = document.getElementById('val-mode-a');
        DOM.valModeB = document.getElementById('val-mode-b');
        DOM.cardModeA = document.getElementById('protocol-mode-a-label');
        DOM.cardModeB = document.getElementById('protocol-mode-b-label');
        DOM.optionAContainer = document.getElementById('option-a-container');
        DOM.optionBContainer = document.getElementById('option-b-container');
        DOM.retrainYes = document.getElementById('retrain-yes');
        DOM.retrainNo = document.getElementById('retrain-no');
        DOM.infoTrainPct = document.getElementById('info-train-pct');
        DOM.infoTestPct = document.getElementById('info-test-pct');
        DOM.infoTestPct2 = document.getElementById('info-test-pct-2');
        DOM.optBTrainSummary = document.getElementById('opt-b-train-summary');
        DOM.optBTestBadge = document.getElementById('opt-b-test-badge');
        DOM.optBTestSummary = document.getElementById('opt-b-test-summary');
        DOM.testDropzone = document.getElementById('test-dropzone');
        DOM.testFileInput = document.getElementById('test-csv-file-input');
        DOM.loadSampleTestBtn = document.getElementById('load-sample-test-btn');

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

        // =========================================================================
        // MODEL TESTING SCOPE & SELECTION (Single, Multi, All Models)
        // =========================================================================
        function setModelScope(scope) {
            AppState.modelScope = scope;
            if (DOM.scopeSingle) DOM.scopeSingle.checked = (scope === 'single');
            if (DOM.scopeMulti) DOM.scopeMulti.checked = (scope === 'multi');
            if (DOM.scopeAll) DOM.scopeAll.checked = (scope === 'all');

            if (DOM.scopeSingleLabel) DOM.scopeSingleLabel.classList.toggle('active-single', scope === 'single');
            if (DOM.scopeMultiLabel) DOM.scopeMultiLabel.classList.toggle('active-multi', scope === 'multi');
            if (DOM.scopeAllLabel) DOM.scopeAllLabel.classList.toggle('active-all', scope === 'all');

            if (scope === 'single') {
                const singleId = AppState.selectedModelId || AppState.selectedModelIds[0] || 'nn';
                AppState.selectedModelIds = [singleId];
                AppState.selectedModelId = singleId;
                AppState.activeHyperparamModelId = singleId;
            } else if (scope === 'all') {
                AppState.selectedModelIds = ALL_MODEL_IDS.slice();
            } else if (scope === 'multi') {
                if (AppState.selectedModelIds.length <= 1) {
                    AppState.selectedModelIds = ['ensemble', 'rf', 'nn'];
                }
            }
            syncModelCardSelections();
            updateTrainButtonLabel();
        }

        function syncModelCardSelections() {
            DOM.modelCards.forEach(card => {
                const mId = card.dataset.modelId;
                const isSelected = AppState.selectedModelIds.includes(mId);
                card.classList.toggle('selected', isSelected);
                const chk = card.querySelector('.model-select-checkbox');
                if (chk) chk.checked = isSelected;
            });

            if (DOM.activeHyperparamSelect) {
                if (AppState.selectedModelIds.includes(AppState.activeHyperparamModelId)) {
                    DOM.activeHyperparamSelect.value = AppState.activeHyperparamModelId;
                } else if (AppState.selectedModelIds.length > 0) {
                    AppState.activeHyperparamModelId = AppState.selectedModelIds[0];
                    DOM.activeHyperparamSelect.value = AppState.activeHyperparamModelId;
                }
            }
            updateHyperparamsVisibility(AppState.activeHyperparamModelId);
        }

        function toggleModelSelection(mId) {
            if (AppState.modelScope === 'single') {
                AppState.selectedModelId = mId;
                AppState.selectedModelIds = [mId];
                AppState.activeHyperparamModelId = mId;
            } else {
                const idx = AppState.selectedModelIds.indexOf(mId);
                if (idx >= 0) {
                    if (AppState.selectedModelIds.length > 1) {
                        AppState.selectedModelIds.splice(idx, 1);
                    } else {
                        // Prevent deselecting last remaining model
                        return;
                    }
                } else {
                    AppState.selectedModelIds.push(mId);
                }
                AppState.activeHyperparamModelId = mId;

                if (AppState.selectedModelIds.length === ALL_MODEL_IDS.length) {
                    AppState.modelScope = 'all';
                } else if (AppState.selectedModelIds.length === 1) {
                    AppState.modelScope = 'single';
                    AppState.selectedModelId = AppState.selectedModelIds[0];
                } else {
                    AppState.modelScope = 'multi';
                }
                if (DOM.scopeSingle) DOM.scopeSingle.checked = (AppState.modelScope === 'single');
                if (DOM.scopeMulti) DOM.scopeMulti.checked = (AppState.modelScope === 'multi');
                if (DOM.scopeAll) DOM.scopeAll.checked = (AppState.modelScope === 'all');
                if (DOM.scopeSingleLabel) DOM.scopeSingleLabel.classList.toggle('active-single', AppState.modelScope === 'single');
                if (DOM.scopeMultiLabel) DOM.scopeMultiLabel.classList.toggle('active-multi', AppState.modelScope === 'multi');
                if (DOM.scopeAllLabel) DOM.scopeAllLabel.classList.toggle('active-all', AppState.modelScope === 'all');
            }
            syncModelCardSelections();
            updateTrainButtonLabel();
        }

        function updateTrainButtonLabel() {
            if (!DOM.trainModelBtn) return;
            const count = AppState.selectedModelIds.length;
            if (count === 1) {
                const mInfo = MODEL_INFO[AppState.selectedModelIds[0]] || { short: AppState.selectedModelIds[0].toUpperCase() };
                DOM.trainModelBtn.textContent = `🚀 Train & Test 1 Model (${mInfo.short})`;
            } else if (count === ALL_MODEL_IDS.length) {
                DOM.trainModelBtn.textContent = `🏆 Run Full Tournament (Train & Test All 6 Models)`;
            } else {
                DOM.trainModelBtn.textContent = `🚀 Train & Benchmark ${count} Selected Models`;
            }
        }

        // Scope Radio Changes & Label Clicks
        if (DOM.scopeSingle) DOM.scopeSingle.addEventListener('change', () => setModelScope('single'));
        if (DOM.scopeMulti) DOM.scopeMulti.addEventListener('change', () => setModelScope('multi'));
        if (DOM.scopeAll) DOM.scopeAll.addEventListener('change', () => setModelScope('all'));
        if (DOM.scopeSingleLabel) DOM.scopeSingleLabel.addEventListener('click', () => setModelScope('single'));
        if (DOM.scopeMultiLabel) DOM.scopeMultiLabel.classList.remove('active-multi');
        if (DOM.scopeMultiLabel) DOM.scopeMultiLabel.addEventListener('click', () => setModelScope('multi'));
        if (DOM.scopeAllLabel) DOM.scopeAllLabel.addEventListener('click', () => setModelScope('all'));

        // Preset Quick Action Buttons
        if (DOM.btnSelectAllModels) {
            DOM.btnSelectAllModels.addEventListener('click', () => setModelScope('all'));
        }
        if (DOM.btnSelectTopModels) {
            DOM.btnSelectTopModels.addEventListener('click', () => {
                AppState.modelScope = 'multi';
                AppState.selectedModelIds = ['ensemble', 'rf', 'nn'];
                if (DOM.scopeMulti) DOM.scopeMulti.checked = true;
                if (DOM.scopeSingleLabel) DOM.scopeSingleLabel.classList.remove('active-single');
                if (DOM.scopeMultiLabel) DOM.scopeMultiLabel.classList.add('active-multi');
                if (DOM.scopeAllLabel) DOM.scopeAllLabel.classList.remove('active-all');
                syncModelCardSelections();
                updateTrainButtonLabel();
            });
        }
        if (DOM.btnClearModelSelection) {
            DOM.btnClearModelSelection.addEventListener('click', () => setModelScope('single'));
        }

        // Model Card Clicks & Checkbox Changes
        DOM.modelCards.forEach(card => {
            const mId = card.dataset.modelId;
            const chk = card.querySelector('.model-select-checkbox');

            card.addEventListener('click', (e) => {
                if (e.target.closest('.model-seed-box')) return;
                if (e.target === chk) return; // checkbox change will handle it
                toggleModelSelection(mId);
            });

            if (chk) {
                chk.addEventListener('change', (e) => {
                    e.stopPropagation();
                    toggleModelSelection(mId);
                });
            }
        });

        // Hyperparameter model selector dropdown
        if (DOM.activeHyperparamSelect) {
            DOM.activeHyperparamSelect.addEventListener('change', (e) => {
                AppState.activeHyperparamModelId = e.target.value;
                updateHyperparamsVisibility(e.target.value);
            });
        }

        // Protocol Mode Toggle (Option A vs Option B)
        function setValidationMode(mode) {
            AppState.validationMode = mode;
            if (mode === 'split') {
                if (DOM.valModeA) DOM.valModeA.checked = true;
                if (DOM.cardModeA) {
                    DOM.cardModeA.classList.add('selected-a');
                    DOM.cardModeA.style.borderColor = 'var(--accent-cyan)';
                    DOM.cardModeA.style.background = 'rgba(6, 182, 212, 0.08)';
                }
                if (DOM.cardModeB) {
                    DOM.cardModeB.classList.remove('selected-b');
                    DOM.cardModeB.style.borderColor = 'var(--border-color)';
                    DOM.cardModeB.style.background = 'var(--bg-surface-subtle)';
                }
                if (DOM.optionAContainer) DOM.optionAContainer.style.display = 'block';
                if (DOM.optionBContainer) DOM.optionBContainer.style.display = 'none';
            } else {
                if (DOM.valModeB) DOM.valModeB.checked = true;
                if (DOM.cardModeB) {
                    DOM.cardModeB.classList.add('selected-b');
                    DOM.cardModeB.style.borderColor = 'var(--accent-purple)';
                    DOM.cardModeB.style.background = 'rgba(168, 85, 247, 0.08)';
                }
                if (DOM.cardModeA) {
                    DOM.cardModeA.classList.remove('selected-a');
                    DOM.cardModeA.style.borderColor = 'var(--border-color)';
                    DOM.cardModeA.style.background = 'var(--bg-surface-subtle)';
                }
                if (DOM.optionAContainer) DOM.optionAContainer.style.display = 'none';
                if (DOM.optionBContainer) DOM.optionBContainer.style.display = 'block';
                refreshOptionBSummaries();
            }
        }

        if (DOM.valModeA) DOM.valModeA.addEventListener('change', () => setValidationMode('split'));
        if (DOM.valModeB) DOM.valModeB.addEventListener('change', () => setValidationMode('separate'));
        if (DOM.cardModeA) DOM.cardModeA.addEventListener('click', () => setValidationMode('split'));
        if (DOM.cardModeB) DOM.cardModeB.addEventListener('click', () => setValidationMode('separate'));

        // Option B Testing Cohort Upload
        if (DOM.testDropzone && DOM.testFileInput) {
            DOM.testDropzone.addEventListener('dragover', (e) => {
                e.preventDefault();
                DOM.testDropzone.classList.add('drag-active');
            });
            DOM.testDropzone.addEventListener('dragleave', () => DOM.testDropzone.classList.remove('drag-active'));
            DOM.testDropzone.addEventListener('drop', (e) => {
                e.preventDefault();
                DOM.testDropzone.classList.remove('drag-active');
                if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                    handleTestFileUpload(e.dataTransfer.files[0]);
                }
            });
            DOM.testDropzone.addEventListener('click', () => DOM.testFileInput.click());
            DOM.testFileInput.addEventListener('change', (e) => {
                if (e.target.files && e.target.files[0]) {
                    handleTestFileUpload(e.target.files[0]);
                }
            });
        }

        if (DOM.loadSampleTestBtn) {
            DOM.loadSampleTestBtn.addEventListener('click', loadSampleTestingCohort);
        }

        // Train / Test Split interactive synchronization (Option A)
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

        // Reproducibility Seed Management (Default 42 vs Custom Seed)
        setupSeedManagement();

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
        const infoTest2 = document.getElementById('info-test-pct-2');

        if (slider) slider.value = train;
        if (trainInput) trainInput.value = train;
        if (testInput) testInput.value = test;
        if (dispTrain) dispTrain.innerText = `${train}% Train`;
        if (dispTest) dispTest.innerText = `${test}% Test`;
        if (infoTrain) infoTrain.innerText = `${train}%`;
        if (infoTest) infoTest.innerText = `${test}%`;
        if (infoTest2) infoTest2.innerText = `${test}%`;
    }

    // Option B File & Sample Handlers
    function handleTestFileUpload(file) {
        if (!file.name.endsWith('.csv') && file.type !== 'text/csv') {
            alert("Please upload a valid .csv file for the testing cohort.");
            return;
        }
        const reader = new FileReader();
        reader.onload = (e) => {
            processTestingCSVContent(e.target.result, file.name);
        };
        reader.readAsText(file);
    }

    function processTestingCSVContent(csvText, fileName) {
        try {
            const parsed = DataEngine.parseCSV(csvText);
            const rows = parsed && parsed.rows ? parsed.rows : null;
            if (!rows || rows.length === 0) {
                alert("The uploaded testing CSV appears to be empty or unparseable.");
                return;
            }
            AppState.separateTestRawRows = rows;
            AppState.separateTestFileName = fileName;
            refreshOptionBSummaries();
        } catch (err) {
            alert("Failed to parse testing CSV: " + err.message);
        }
    }

    async function loadSampleTestingCohort() {
        try {
            let filename = 'medical_batch_unseen_cases.csv';
            let label = 'Neurosurgical External Validation Cohort';
            if (AppState.fileName && AppState.fileName.includes('recovery')) {
                filename = 'medical_recovery_test_cohort.csv';
                label = 'Functional Recovery Test Cohort';
            } else if (AppState.fileName && (AppState.fileName.includes('universal') || AppState.fileName.includes('cardio'))) {
                filename = 'universal_medical_test_cohort.csv';
                label = 'Cardiometabolic Risk Test Cohort';
            }

            const resp = await fetch(filename);
            if (!resp.ok) throw new Error("Could not load " + filename);
            const text = await resp.text();
            processTestingCSVContent(text, `${label} (${filename})`);
        } catch (err) {
            console.error("Error loading sample test cohort:", err);
            alert("Could not load sample test cohort: " + err.message);
        }
    }

    function refreshOptionBSummaries() {
        if (!DOM.optBTrainSummary) return;

        // Dataset (i) Summary
        if (AppState.imputedData && AppState.columnConfigs) {
            const targetCol = Object.keys(AppState.columnConfigs).find(c => AppState.columnConfigs[c].role === 'target');
            const targetCfg = targetCol ? AppState.columnConfigs[targetCol] : null;
            const isClf = targetCfg ? (targetCfg.type === 'binary' || targetCfg.type === 'multiclass') : true;
            DOM.optBTrainSummary.innerHTML = `
                <div><strong>Source:</strong> ${AppState.fileName || 'Active Dataset'}</div>
                <div><strong>Training Cases:</strong> ${AppState.imputedData.imputedRows.length} patients (100% used for model parameter fitting)</div>
                <div><strong>Target:</strong> <span style="color: var(--accent-cyan); font-weight: 700;">${targetCol || 'Unspecified'}</span> (${isClf ? 'Classification' : 'Continuous Regression'})</div>
            `;
        } else {
            DOM.optBTrainSummary.innerHTML = `
                <span style="color: var(--text-muted);">Please complete Steps 1-3 first to finalize the training cohort.</span>
            `;
        }

        // Dataset (ii) Summary
        if (AppState.separateTestRawRows) {
            const testValidCount = AppState.separateTestRawRows.length;
            let targetStatus = "Checking schema...";
            if (AppState.columnConfigs) {
                const targetCol = Object.keys(AppState.columnConfigs).find(c => AppState.columnConfigs[c].role === 'target');
                if (targetCol) {
                    const hasTarget = AppState.separateTestRawRows[0] && (targetCol in AppState.separateTestRawRows[0]);
                    targetStatus = hasTarget 
                        ? `<span style="color: var(--accent-emerald); font-weight: 600;">✓ Target '${targetCol}' found</span>`
                        : `<span style="color: var(--accent-amber); font-weight: 600;">⚠️ Target '${targetCol}' not in header!</span>`;
                }
            }

            if (DOM.optBTestBadge) {
                DOM.optBTestBadge.className = 'pill pill-target';
                DOM.optBTestBadge.style.background = 'rgba(168, 85, 247, 0.15)';
                DOM.optBTestBadge.style.color = 'var(--accent-purple)';
                DOM.optBTestBadge.style.borderColor = 'rgba(168, 85, 247, 0.4)';
                DOM.optBTestBadge.innerText = `✓ Ready (${testValidCount} cases)`;
            }

            if (DOM.optBTestSummary) {
                DOM.optBTestSummary.innerHTML = `
                    <div style="background: var(--bg-surface); padding: 10px 12px; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle); display: flex; flex-direction: column; gap: 4px;">
                        <div><strong>Testing File:</strong> ${AppState.separateTestFileName || 'Testing Cohort CSV'}</div>
                        <div><strong>Total Test Cases:</strong> ${testValidCount} external patients</div>
                        <div><strong>Target Verification:</strong> ${targetStatus}</div>
                        <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">
                            🔒 Test data will be transformed strictly using training metadata with zero data leakage.
                        </div>
                    </div>
                `;
            }
        } else {
            if (DOM.optBTestBadge) {
                DOM.optBTestBadge.className = 'pill pill-meta';
                DOM.optBTestBadge.style.background = '';
                DOM.optBTestBadge.style.color = '';
                DOM.optBTestBadge.style.borderColor = '';
                DOM.optBTestBadge.innerText = 'Pending Upload';
            }
            if (DOM.optBTestSummary) {
                DOM.optBTestSummary.innerHTML = `
                    <span style="color: var(--text-muted); font-size: 11.5px;">No separate testing dataset provided yet. Upload a CSV above or click 'Load Built-In Validation Cohort'.</span>
                `;
            }
        }
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
        if (stepNum === 4) refreshOptionBSummaries();
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
    function updateHyperparamsVisibility(targetModelId) {
        const activeId = targetModelId || AppState.activeHyperparamModelId || AppState.selectedModelId || 'nn';
        AppState.activeHyperparamModelId = activeId;
        const isNN = activeId === 'nn';
        const isRF = activeId === 'rf';
        const isGLM = activeId === 'glm';
        const isKNN = activeId === 'knn';
        const isNB = activeId === 'nb';
        const isEnsemble = activeId === 'ensemble';

        const nnBox = document.getElementById('nn-params-box');
        const rfBox = document.getElementById('rf-params-box');
        const glmBox = document.getElementById('glm-params-box');
        const knnBox = document.getElementById('knn-params-box');
        const nbBox = document.getElementById('nb-params-box');
        const ensembleBox = document.getElementById('ensemble-params-box');

        if (nnBox) nnBox.style.display = isNN ? 'grid' : 'none';
        if (rfBox) rfBox.style.display = isRF ? 'grid' : 'none';
        if (glmBox) glmBox.style.display = isGLM ? 'grid' : 'none';
        if (knnBox) knnBox.style.display = isKNN ? 'grid' : 'none';
        if (nbBox) nbBox.style.display = isNB ? 'grid' : 'none';
        if (ensembleBox) ensembleBox.style.display = isEnsemble ? 'grid' : 'none';

        if (DOM.activeHyperparamSelect && DOM.activeHyperparamSelect.value !== activeId) {
            DOM.activeHyperparamSelect.value = activeId;
        }
    }

    // =========================================================================
    // REPRODUCIBILITY SEED MANAGEMENT (DEFAULT 42 VS CUSTOM SEED)
    // =========================================================================
    function setupSeedManagement() {
        const ALL_MODELS = ['nn', 'rf', 'glm', 'knn', 'nb', 'ensemble'];

        // Helper to update a model's seed mode (default 42 vs custom)
        function setModelSeedMode(mId, mode) {
            const cardDefaultRadio = document.querySelector(`input[name="seed-mode-${mId}"][value="default"]`);
            const cardCustomRadio = document.querySelector(`input[name="seed-mode-${mId}"][value="custom"]`);
            const hpDefaultRadio = document.querySelector(`input[name="hp-seed-mode-${mId}"][value="default"]`);
            const hpCustomRadio = document.querySelector(`input[name="hp-seed-mode-${mId}"][value="custom"]`);
            const cardCustomRow = document.getElementById(`custom-seed-row-${mId}`);
            const hpCustomWrap = document.getElementById(`hp-seed-wrap-${mId}`);
            const cardInput = document.getElementById(`${mId}-card-seed-input`);
            const hpInput = document.getElementById(`${mId}-seed-input`);

            if (mode === 'default') {
                if (cardDefaultRadio) cardDefaultRadio.checked = true;
                if (hpDefaultRadio) hpDefaultRadio.checked = true;
                if (cardCustomRow) cardCustomRow.style.display = 'none';
                if (hpCustomWrap) hpCustomWrap.style.display = 'none';
                if (cardInput) cardInput.value = 42;
                if (hpInput) hpInput.value = 42;
            } else {
                if (cardCustomRadio) cardCustomRadio.checked = true;
                if (hpCustomRadio) hpCustomRadio.checked = true;
                if (cardCustomRow) cardCustomRow.style.display = 'flex';
                if (hpCustomWrap) hpCustomWrap.style.display = 'flex';
                if (cardInput && (!cardInput.value || cardInput.value === '42')) cardInput.focus();
                else if (hpInput) hpInput.focus();
            }
        }

        // Attach listeners for Card Seed Radios and Hyperparam Box Seed Radios
        ALL_MODELS.forEach(mId => {
            const cardRadios = document.querySelectorAll(`input[name="seed-mode-${mId}"]`);
            cardRadios.forEach(radio => {
                radio.addEventListener('change', (e) => {
                    setModelSeedMode(mId, e.target.value);
                });
            });

            const hpRadios = document.querySelectorAll(`input[name="hp-seed-mode-${mId}"]`);
            hpRadios.forEach(radio => {
                radio.addEventListener('change', (e) => {
                    setModelSeedMode(mId, e.target.value);
                });
            });

            // Two-way sync for custom seed inputs
            const cardInput = document.getElementById(`${mId}-card-seed-input`);
            const hpInput = document.getElementById(`${mId}-seed-input`);

            if (cardInput && hpInput) {
                cardInput.addEventListener('input', (e) => {
                    hpInput.value = e.target.value;
                });
                hpInput.addEventListener('input', (e) => {
                    cardInput.value = e.target.value;
                });
            }

            // Dice roll buttons for Card
            const cardDice = document.querySelector(`.card-btn-roll-seed[data-model="${mId}"]`);
            if (cardDice) {
                cardDice.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const newSeed = Math.floor(Math.random() * 90000) + 10000;
                    if (cardInput) cardInput.value = newSeed;
                    if (hpInput) hpInput.value = newSeed;
                    showSeedToast(`Generated ${MODEL_INFO[mId]?.name || mId} Seed: ${newSeed}`);
                });
            }

            // Dice roll buttons for Hyperparam Box
            const hpDice = document.getElementById(`btn-roll-${mId}-seed`);
            if (hpDice) {
                hpDice.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const newSeed = Math.floor(Math.random() * 90000) + 10000;
                    if (cardInput) cardInput.value = newSeed;
                    if (hpInput) hpInput.value = newSeed;
                    showSeedToast(`Generated ${MODEL_INFO[mId]?.name || mId} Seed: ${newSeed}`);
                });
            }
        });

        // Split Seed (Option A) Controls
        const splitRadios = document.querySelectorAll('input[name="split-seed-mode"]');
        const splitCustomWrap = document.getElementById('split-seed-custom-wrap');
        const splitInput = document.getElementById('split-seed-input');
        const splitDice = document.getElementById('btn-roll-split-seed');

        splitRadios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                if (e.target.value === 'default') {
                    if (splitCustomWrap) splitCustomWrap.style.display = 'none';
                    if (splitInput) splitInput.value = 42;
                } else {
                    if (splitCustomWrap) splitCustomWrap.style.display = 'flex';
                    if (splitInput) splitInput.focus();
                }
            });
        });

        if (splitDice && splitInput) {
            splitDice.addEventListener('click', () => {
                const newSeed = Math.floor(Math.random() * 90000) + 10000;
                splitInput.value = newSeed;
                splitInput.focus();
                showSeedToast(`Generated Split Seed: ${newSeed}`);
            });
        }

        // Quick Global Seed Action Buttons
        const btnAllDefault = document.getElementById('btn-all-default-seed');
        if (btnAllDefault) {
            btnAllDefault.addEventListener('click', () => {
                ALL_MODELS.forEach(mId => setModelSeedMode(mId, 'default'));
                const defSplitRadio = document.getElementById('split-seed-mode-default');
                if (defSplitRadio) {
                    defSplitRadio.checked = true;
                    if (splitCustomWrap) splitCustomWrap.style.display = 'none';
                    if (splitInput) splitInput.value = 42;
                }
                showSeedToast('All models set to Default Seed (42)');
            });
        }

        const btnAllRandom = document.getElementById('btn-all-random-seed');
        if (btnAllRandom) {
            btnAllRandom.addEventListener('click', () => {
                ALL_MODELS.forEach(mId => {
                    setModelSeedMode(mId, 'custom');
                    const randSeed = Math.floor(Math.random() * 90000) + 10000;
                    const cInp = document.getElementById(`${mId}-card-seed-input`);
                    const hInp = document.getElementById(`${mId}-seed-input`);
                    if (cInp) cInp.value = randSeed;
                    if (hInp) hInp.value = randSeed;
                });
                showSeedToast('All models assigned individual random seeds 🎲');
            });
        }
    }

    function getHyperparamsForModel(modelId, prep) {
        const hp = {};
        if (modelId === 'nn') {
            hp.epochs = parseInt(document.getElementById('nn-epochs-input')?.value, 10) || 50;
            hp.learningRate = parseFloat(document.getElementById('nn-lr-input')?.value) || 0.005;
            const h1 = parseInt(document.getElementById('nn-h1-input')?.value, 10) || 64;
            const h2 = parseInt(document.getElementById('nn-h2-input')?.value, 10) || 32;
            const h3El = document.getElementById('nn-h3-input');
            const h3 = h3El ? (parseInt(h3El.value, 10) || 0) : 0;
            hp.hiddenLayers = h3 > 0 ? [h1, h2, h3] : [h1, h2];
            hp.dropoutRate = parseFloat(document.getElementById('nn-dropout-input')?.value) || 0.1;
            hp.activation = document.getElementById('nn-act-select')?.value || 'relu';
            const optEl = document.getElementById('nn-optimizer-select');
            hp.optimizer = optEl ? optEl.value : 'adam';
            const batchEl = document.getElementById('nn-batch-select');
            hp.batchSize = batchEl ? (parseInt(batchEl.value, 10) || 16) : 16;
            const l2El = document.getElementById('nn-l2-select');
            hp.l2Reg = l2El ? parseFloat(l2El.value) : 0.0001;
        } else if (modelId === 'rf') {
            hp.numTrees = parseInt(document.getElementById('rf-trees-input')?.value, 10) || 30;
            hp.maxDepth = parseInt(document.getElementById('rf-depth-input')?.value, 10) || 6;
            const minSplitEl = document.getElementById('rf-minsplit-select');
            hp.minSamplesSplit = minSplitEl ? (parseInt(minSplitEl.value, 10) || 2) : 2;
            const maxFeatEl = document.getElementById('rf-maxfeatures-select');
            hp.featureRatio = maxFeatEl ? maxFeatEl.value : 'sqrt';
        } else if (modelId === 'glm') {
            hp.lr = parseFloat(document.getElementById('glm-lr-input')?.value) || 0.05;
            hp.epochs = parseInt(document.getElementById('glm-epochs-input')?.value, 10) || 150;
            const lambdaEl = document.getElementById('glm-lambda-select');
            hp.lambdaL2 = lambdaEl ? parseFloat(lambdaEl.value) : 0.01;
            const solverEl = document.getElementById('glm-solver-select');
            hp.solver = solverEl ? solverEl.value : 'momentum';
        } else if (modelId === 'knn') {
            hp.k = parseInt(document.getElementById('knn-k-input')?.value, 10) || 5;
            const metricEl = document.getElementById('knn-metric-select');
            hp.metric = metricEl ? metricEl.value : 'euclidean';
            const weightEl = document.getElementById('knn-weighting-select');
            hp.weighting = weightEl ? weightEl.value : 'uniform';
        } else if (modelId === 'nb') {
            const smoothEl = document.getElementById('nb-smoothing-select');
            hp.varSmoothing = smoothEl ? parseFloat(smoothEl.value) : 1e-4;
            const priorsEl = document.getElementById('nb-priors-select');
            hp.priorMode = priorsEl ? priorsEl.value : 'empirical';
        } else if (modelId === 'ensemble') {
            const selectedBases = [];
            if (document.getElementById('ens-include-nn')?.checked) selectedBases.push('nn');
            if (document.getElementById('ens-include-rf')?.checked) selectedBases.push('rf');
            if (document.getElementById('ens-include-glm')?.checked) selectedBases.push('glm');
            if (document.getElementById('ens-include-knn')?.checked) selectedBases.push('knn');
            if (document.getElementById('ens-include-nb')?.checked && prep?.isClassification) selectedBases.push('nb');
            if (selectedBases.length === 0) selectedBases.push('rf', 'glm');

            hp.voting = document.getElementById('ensemble-voting-select')?.value || 'soft';
            hp.baseModels = selectedBases;
            const powerEl = document.getElementById('ens-power-select');
            hp.power = powerEl ? parseFloat(powerEl.value) : 2;
            const capEl = document.getElementById('ens-capacity-select');
            hp.capacity = capEl ? capEl.value : 'balanced';
        }

        // Extract Reproducibility Seed for this model (Default 42 vs Custom Seed)
        const cardModeRadio = document.querySelector(`input[name="seed-mode-${modelId}"]:checked`);
        const hpModeRadio = document.querySelector(`input[name="hp-seed-mode-${modelId}"]:checked`);
        const isDefault = (cardModeRadio ? cardModeRadio.value === 'default' : true) &&
                          (hpModeRadio ? hpModeRadio.value === 'default' : true);

        let seedVal = 42;
        if (!isDefault) {
            const cardInput = document.getElementById(`${modelId}-card-seed-input`);
            const hpInput = document.getElementById(`${modelId}-seed-input`);
            const inputVal = (cardInput && cardInput.value !== '') ? cardInput.value : (hpInput ? hpInput.value : '');
            if (inputVal !== '' && !isNaN(parseInt(inputVal, 10))) {
                seedVal = parseInt(inputVal, 10);
            } else {
                seedVal = 42;
            }
        }
        hp.seed = seedVal;

        return hp;
    }

    async function startModelTraining() {
        if (!AppState.imputedData) {
            alert("Please execute Data Imputation first.");
            return;
        }

        try {
            DOM.trainModelBtn.disabled = true;
            DOM.trainStatus.innerHTML = `<span style="color: var(--accent-cyan); display: inline-flex; align-items: center; gap: 6px;">⚙️ Initializing multi-model tournament and data partitions...</span>`;

            // Prepare features and target from training cohort
            const prep = DataEngine.prepareFeaturesAndTarget(AppState.imputedData.imputedRows, AppState.columnConfigs);
            AppState.prepMeta = prep;

            // Determine active validation protocol (Option A vs Option B)
            const isOptionA = DOM.valModeA ? DOM.valModeA.checked : (AppState.validationMode !== 'separate');

            // Determine models to test
            let modelsToTest = (AppState.selectedModelIds && AppState.selectedModelIds.length > 0)
                ? AppState.selectedModelIds.slice()
                : [AppState.selectedModelId || 'nn'];

            // Continuous regression guard: Naive Bayes is classification-only
            if (!prep.isClassification) {
                modelsToTest = modelsToTest.filter(mId => mId !== 'nb');
            }
            if (modelsToTest.length === 0) modelsToTest = ['nn'];

            const testedResults = [];

            if (isOptionA) {
                // =============================================================
                // OPTION A: INTERNAL TRAIN / TEST SPLIT WITH RETRAINING CHOICE
                // =============================================================
                const trainPctInput = document.getElementById('train-pct-input');
                const trainPct = trainPctInput ? parseFloat(trainPctInput.value) : 80;
                const trainRatio = Math.min(0.95, Math.max(0.50, (trainPct || 80) / 100));
                const testPct = Math.round((1 - trainRatio) * 100);

                // Option A Reproducibility Seed for train/test split (Default 42 vs Custom Seed)
                const splitModeRadio = document.querySelector('input[name="split-seed-mode"]:checked');
                const isSplitDefault = splitModeRadio ? splitModeRadio.value === 'default' : true;
                let splitSeed = 42;
                if (!isSplitDefault) {
                    const splitSeedInput = document.getElementById('split-seed-input');
                    if (splitSeedInput && splitSeedInput.value !== '' && !isNaN(parseInt(splitSeedInput.value, 10))) {
                        splitSeed = parseInt(splitSeedInput.value, 10);
                    } else {
                        splitSeed = 42;
                    }
                }

                const split = DataEngine.trainTestSplit(prep.X, prep.y, trainRatio, prep.isClassification, splitSeed);
                AppState.trainTestData = split;

                const shouldRetrain = DOM.retrainYes ? DOM.retrainYes.checked : true;

                for (let i = 0; i < modelsToTest.length; i++) {
                    const mId = modelsToTest[i];
                    const mInfo = MODEL_INFO[mId] || { name: mId.toUpperCase(), icon: '⚙️' };
                    const hyperparams = getHyperparamsForModel(mId, prep);

                    DOM.trainStatus.innerHTML = `<span style="color: var(--accent-cyan);">🚀 [${i + 1}/${modelsToTest.length}] Training ${mInfo.name} on ${Math.round(trainRatio * 100)}% split (${split.trainX.length} cases &bull; Seed: ${hyperparams.seed})...</span>`;
                    await new Promise(r => setTimeout(r, 40));

                    if (modelsToTest.length === 1 && mId === 'nn') {
                        setupTrainingLossChart();
                    }

                    const valModel = ModelEngine.createModel(
                        mId,
                        prep.isClassification,
                        prep.targetClasses.length || 2,
                        hyperparams
                    );

                    if (mId === 'nn') {
                        await valModel.fit(split.trainX, split.trainY, split.testX, split.testY, (epoch, total, logs) => {
                            DOM.trainStatus.innerHTML = `<span style="color: var(--accent-cyan);">[${i + 1}/${modelsToTest.length}] ${mInfo.name}: Epoch ${epoch}/${total} &mdash; Train Loss: ${logs.loss.toFixed(4)} ${logs.val_loss !== undefined ? `| Test Loss: ${logs.val_loss.toFixed(4)}` : ''}</span>`;
                            if (modelsToTest.length === 1) updateLossChart(epoch, logs.loss, logs.val_loss);
                        });
                    } else if (mId === 'ensemble') {
                        await valModel.fit(split.trainX, split.trainY, split.testX, split.testY, (curr, total, info) => {
                            DOM.trainStatus.innerHTML = `<span style="color: var(--accent-cyan);">[${i + 1}/${modelsToTest.length}] ${mInfo.name}: Sub-Model ${curr}/${total} (${info.stage || ''})...</span>`;
                        });
                    } else {
                        valModel.fit(split.trainX, split.trainY);
                    }

                    // Benchmark out-of-sample accuracy on held-out test split
                    const evalRes = ModelEngine.evaluate(valModel, split.testX, split.testY, prep.isClassification, prep.targetClasses);
                    evalRes.protocol = 'option-a';
                    evalRes.retrainOn100 = shouldRetrain;
                    evalRes.trainRatio = trainRatio;
                    evalRes.trainPct = Math.round(trainRatio * 100);
                    evalRes.testPct = testPct;
                    evalRes.trainCount = split.trainX.length;
                    evalRes.testCount = split.testX.length;
                    evalRes.totalCount = prep.X.length;
                    evalRes.splitSeed = splitSeed;
                    evalRes.seed = hyperparams.seed;

                    let activeModelInstance = valModel;

                    if (shouldRetrain) {
                        DOM.trainStatus.innerHTML = `<span style="color: var(--accent-amber); font-weight: 600;">✓ [${i + 1}/${modelsToTest.length}] ${mInfo.name} benchmarked. Retraining on 100% of cohort (N=${prep.X.length})...</span>`;
                        await new Promise(r => setTimeout(r, 40));

                        const fullModel = ModelEngine.createModel(
                            mId,
                            prep.isClassification,
                            prep.targetClasses.length || 2,
                            hyperparams
                        );

                        if (mId === 'nn') {
                            await fullModel.fit(prep.X, prep.y, null, null, (epoch, total, logs) => {
                                DOM.trainStatus.innerHTML = `<span style="color: var(--accent-cyan);">[${i + 1}/${modelsToTest.length}] ${mInfo.name} 100% Retrain: Epoch ${epoch}/${total} &mdash; Loss: ${logs.loss.toFixed(4)}</span>`;
                            });
                        } else if (mId === 'ensemble') {
                            await fullModel.fit(prep.X, prep.y, null, null, (curr, total, info) => {
                                DOM.trainStatus.innerHTML = `<span style="color: var(--accent-cyan);">[${i + 1}/${modelsToTest.length}] ${mInfo.name} 100% Retrain: Base ${curr}/${total}...</span>`;
                            });
                        } else {
                            fullModel.fit(prep.X, prep.y);
                        }
                        activeModelInstance = fullModel;
                    }

                    testedResults.push({
                        modelId: mId,
                        info: mInfo,
                        evalRes: evalRes,
                        model: activeModelInstance,
                        retrained: shouldRetrain,
                        seed: hyperparams.seed,
                        splitSeed: splitSeed
                    });
                }
            } else {
                // =============================================================
                // OPTION B: SEPARATE TRAINING (i) AND TESTING (ii) DATASETS
                // =============================================================
                if (!AppState.separateTestRawRows || AppState.separateTestRawRows.length === 0) {
                    DOM.trainStatus.innerHTML = `<span style="color: var(--accent-amber);">⚠️ Option B requires a separate testing cohort (Dataset ii). Please upload a test CSV or click 'Load Built-In Validation Cohort'.</span>`;
                    DOM.trainModelBtn.disabled = false;
                    alert("Option B requires a separate testing cohort (Dataset ii). Please upload a test CSV or load the built-in validation cohort in the Option B card.");
                    return;
                }

                // Transform separate test data using training metadata (strict zero leakage)
                const testPrep = DataEngine.prepareTestingDataset(AppState.separateTestRawRows, prep);
                AppState.separateTestData = testPrep;

                if (!testPrep.totalEvaluated || testPrep.totalEvaluated === 0) {
                    DOM.trainStatus.innerHTML = `<span style="color: var(--accent-rose);">Testing cohort evaluation failed: 0 valid test cases found.</span>`;
                    DOM.trainModelBtn.disabled = false;
                    alert(`Could not extract any valid test cases with matching predictors and target. Missing target cases: ${testPrep.missingTargetCount}, unknown classes: ${testPrep.unknownClassCount}. Please check the testing CSV.`);
                    return;
                }

                for (let i = 0; i < modelsToTest.length; i++) {
                    const mId = modelsToTest[i];
                    const mInfo = MODEL_INFO[mId] || { name: mId.toUpperCase(), icon: '⚙️' };
                    const hyperparams = getHyperparamsForModel(mId, prep);

                    DOM.trainStatus.innerHTML = `<span style="color: var(--accent-purple);">🚀 [${i + 1}/${modelsToTest.length}] Fitting ${mInfo.name} on 100% of Training Dataset (i) (${prep.X.length} cases &bull; Seed: ${hyperparams.seed})...</span>`;
                    await new Promise(r => setTimeout(r, 40));

                    if (modelsToTest.length === 1 && mId === 'nn') {
                        setupTrainingLossChart();
                    }

                    const model = ModelEngine.createModel(
                        mId,
                        prep.isClassification,
                        prep.targetClasses.length || 2,
                        hyperparams
                    );

                    if (mId === 'nn') {
                        await model.fit(prep.X, prep.y, testPrep.X, testPrep.y, (epoch, total, logs) => {
                            DOM.trainStatus.innerHTML = `<span style="color: var(--accent-purple);">[${i + 1}/${modelsToTest.length}] ${mInfo.name}: Epoch ${epoch}/${total} &mdash; Train Loss: ${logs.loss.toFixed(4)} ${logs.val_loss !== undefined ? `| External Test Loss: ${logs.val_loss.toFixed(4)}` : ''}</span>`;
                            if (modelsToTest.length === 1) updateLossChart(epoch, logs.loss, logs.val_loss);
                        });
                    } else if (mId === 'ensemble') {
                        await model.fit(prep.X, prep.y, testPrep.X, testPrep.y, (curr, total, info) => {
                            DOM.trainStatus.innerHTML = `<span style="color: var(--accent-purple);">[${i + 1}/${modelsToTest.length}] ${mInfo.name}: Base ${curr}/${total} (${info.stage || ''})...</span>`;
                        });
                    } else {
                        model.fit(prep.X, prep.y);
                    }

                    // Evaluate on independent Dataset (ii)
                    const evalRes = ModelEngine.evaluate(model, testPrep.X, testPrep.y, prep.isClassification, prep.targetClasses);
                    evalRes.protocol = 'option-b';
                    evalRes.retrainOn100 = false;
                    evalRes.trainCount = prep.X.length;
                    evalRes.testCount = testPrep.totalEvaluated;
                    evalRes.totalCount = prep.X.length + testPrep.totalEvaluated;
                    evalRes.testProvided = testPrep.totalProvided;
                    evalRes.missingTargetCount = testPrep.missingTargetCount;
                    evalRes.unknownClassCount = testPrep.unknownClassCount;
                    evalRes.seed = hyperparams.seed;

                    testedResults.push({
                        modelId: mId,
                        info: mInfo,
                        evalRes: evalRes,
                        model: model,
                        retrained: false,
                        seed: hyperparams.seed
                    });
                }
            }

            // Sort tested models by out-of-sample diagnostic performance (highest accuracy / R² first)
            if (prep.isClassification) {
                testedResults.sort((a, b) => (b.evalRes.accuracy - a.evalRes.accuracy) || ((b.evalRes.rocAuc || 0) - (a.evalRes.rocAuc || 0)));
            } else {
                testedResults.sort((a, b) => (b.evalRes.r2 - a.evalRes.r2) || (a.evalRes.rmse - b.evalRes.rmse));
            }

            AppState.allTestedResults = testedResults;
            // Designate top-performing model as champion
            const champion = testedResults[0];
            AppState.trainedModel = champion.model;
            AppState.deployedModelId = champion.modelId;
            AppState.evalResults = champion.evalRes;
            AppState.selectedModelId = champion.modelId;

            // Update train status banner
            if (testedResults.length === 1) {
                const singleRes = champion.evalRes;
                const accText = singleRes.isClassification
                    ? `Test Accuracy: ${(singleRes.accuracy * 100).toFixed(1)}% (ROC-AUC: ${singleRes.rocAuc !== null ? singleRes.rocAuc.toFixed(3) : 'N/A'})`
                    : `Test R²: ${singleRes.r2.toFixed(3)} (RMSE: ${singleRes.rmse.toFixed(3)})`;
                DOM.trainStatus.innerHTML = `
                    <div style="display: flex; flex-direction: column; gap: 3px;">
                        <span style="color: var(--accent-emerald); font-weight: 700;">
                            ✓ Successfully Evaluated &amp; Deployed ${champion.info.name}!
                        </span>
                        <span style="font-size: 11.5px; color: var(--text-secondary);">
                            Reported Benchmark: <strong>${accText}</strong> &bull; Reproducibility Seed: <code style="color: var(--accent-purple); font-weight: 700; background: rgba(168,85,247,0.12); padding: 1px 6px; border-radius: 4px;">🌱 ${champion.evalRes.seed !== null ? champion.evalRes.seed : 'Auto'}</code> &bull; Active Predictor: <strong>${champion.info.name} (${champion.retrained ? '100% Cohort Retrained' : (isOptionA ? `${singleRes.trainPct}% Split Model` : 'Dataset i Fitted')})</strong>
                        </span>
                    </div>
                `;
            } else {
                const topAccText = champion.evalRes.isClassification
                    ? `Top Accuracy: ${(champion.evalRes.accuracy * 100).toFixed(1)}%`
                    : `Top R²: ${champion.evalRes.r2.toFixed(3)}`;
                DOM.trainStatus.innerHTML = `
                    <div style="display: flex; flex-direction: column; gap: 3px;">
                        <span style="color: var(--accent-emerald); font-weight: 700;">
                            🏆 Tournament Complete: Tested &amp; Evaluated ${testedResults.length} Models!
                        </span>
                        <span style="font-size: 11.5px; color: var(--text-secondary);">
                            Leader: <strong>${champion.info.name}</strong> [${topAccText} &bull; Seed: <code style="color: var(--accent-purple); font-weight: 700;">🌱 ${champion.evalRes.seed !== null ? champion.evalRes.seed : 'Auto'}</code>] &bull; All model seeds preserved for exact reproduction below.
                        </span>
                    </div>
                `;
            }

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

    function deployTestedModel(modelId) {
        if (!AppState.allTestedResults) return;
        const targetItem = AppState.allTestedResults.find(r => r.modelId === modelId);
        if (!targetItem) return;

        AppState.deployedModelId = modelId;
        AppState.trainedModel = targetItem.model;
        AppState.evalResults = targetItem.evalRes;
        AppState.selectedModelId = modelId;

        // Re-render dashboard to refresh badges and active buttons
        renderEvaluationDashboard();

        // Update single prediction form headers if open
        if (AppState.prepMeta) {
            renderSinglePredictionForm();
        }

        const mInfo = MODEL_INFO[modelId] || { name: modelId.toUpperCase() };
        alert(`✓ Model Deployed: "${mInfo.name}" is now the active predictor for Single Prediction (Step 5) and Batch Prediction (Step 6).`);
    }
    window.deployTestedModel = deployTestedModel;

    function showSeedToast(msg) {
        const existing = document.querySelector('.seed-toast');
        if (existing) existing.remove();
        const toast = document.createElement('div');
        toast.className = 'seed-toast';
        toast.innerHTML = `<span>📋</span> <span>${msg}</span>`;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 2600);
    }

    function copySeedToClipboard(seed) {
        if (seed === undefined || seed === null) return;
        const seedStr = String(seed);
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(seedStr).then(() => {
                showSeedToast(`Seed "${seedStr}" copied to clipboard! (Keep as reference)`);
            }).catch(() => {
                prompt("Copy reproducibility seed:", seedStr);
            });
        } else {
            prompt("Copy reproducibility seed:", seedStr);
        }
    }
    window.copySeedToClipboard = copySeedToClipboard;

    function renderEvaluationDashboard() {
        const evalRes = AppState.evalResults;
        const prep = AppState.prepMeta;
        const allResults = AppState.allTestedResults || [];
        DOM.resultsDashboard.style.display = 'flex';

        const isOptionB = evalRes.protocol === 'option-b';
        const trainPct = evalRes.trainPct || 80;
        const testPct = evalRes.testPct || 20;
        const trainCount = evalRes.trainCount || Math.round(evalRes.totalCount * 0.8);
        const testCount = evalRes.testCount || (evalRes.totalCount - trainCount);
        const totalCount = evalRes.totalCount || evalRes.totalSamples;

        // Protocol Status Banner
        if (DOM.retrainStatusBanner) {
            if (isOptionB) {
                DOM.retrainStatusBanner.innerHTML = `
                    <div style="background: rgba(168, 85, 247, 0.08); border: 1px solid rgba(168, 85, 247, 0.3); border-radius: var(--radius-md); padding: 12px 16px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                        <div>
                            <div style="font-weight: 700; font-size: 13.5px; color: var(--accent-purple); display: flex; align-items: center; gap: 6px;">
                                <span>🛡️</span> Option B: Independent External Validation Cohort
                            </div>
                            <div style="font-size: 12px; color: var(--text-secondary); margin-top: 3px;">
                                Tested on separate external <strong>Dataset (ii) (${testCount} test cases)</strong> with zero data leakage. All models trained on 100% of <strong>Dataset (i) (${trainCount} patients)</strong>.
                            </div>
                        </div>
                        <span class="pill" style="background: rgba(168, 85, 247, 0.18); color: var(--accent-purple); border: 1px solid rgba(168, 85, 247, 0.4); font-size: 12px; font-weight: 700;">
                            External Cohort Validated
                        </span>
                    </div>
                `;
            } else if (evalRes.retrainOn100) {
                DOM.retrainStatusBanner.innerHTML = `
                    <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.28); border-radius: var(--radius-md); padding: 12px 16px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                        <div>
                            <div style="font-weight: 700; font-size: 13.5px; color: var(--accent-emerald); display: flex; align-items: center; gap: 6px;">
                                <span>✓</span> Option A: Two-Phase Retraining Protocol &bull; 100% Retrained Models Active
                            </div>
                            <div style="font-size: 12px; color: var(--text-secondary); margin-top: 3px;">
                                Accuracy evaluated on held-out <strong>${testPct}% test split (${testCount} cases)</strong> after fitting on <strong>${trainPct}% (${trainCount} cases)</strong>. Deployed models were then fully retrained on <strong>100% of cohort data (${totalCount} cases)</strong>.
                            </div>
                        </div>
                        <span class="pill pill-target" style="background: rgba(16, 185, 129, 0.18); color: var(--accent-emerald); border-color: rgba(16, 185, 129, 0.4); font-size: 12px; font-weight: 700;">
                            100% Retrained Models Active
                        </span>
                    </div>
                `;
            } else {
                DOM.retrainStatusBanner.innerHTML = `
                    <div style="background: rgba(6, 182, 212, 0.08); border: 1px solid rgba(6, 182, 212, 0.28); border-radius: var(--radius-md); padding: 12px 16px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                        <div>
                            <div style="font-weight: 700; font-size: 13.5px; color: var(--accent-cyan); display: flex; align-items: center; gap: 6px;">
                                <span>ℹ️</span> Option A: Single-Phase Protocol &bull; Split-Trained Models Active
                            </div>
                            <div style="font-size: 12px; color: var(--text-secondary); margin-top: 3px;">
                                Evaluated on held-out <strong>${testPct}% test split (${testCount} cases)</strong>. Models maintained strictly on the <strong>${trainPct}% training split (${trainCount} cases)</strong> without 100% retraining.
                            </div>
                        </div>
                        <span class="pill" style="background: rgba(6, 182, 212, 0.18); color: var(--accent-cyan); border: 1px solid rgba(6, 182, 212, 0.4); font-size: 12px; font-weight: 700;">
                            ${trainPct}% Split Models Active
                        </span>
                    </div>
                `;
            }
        }

        const cohortDesc = isOptionB ? `External Testing Cohort (Dataset ii: ${testCount} Patients)` : `Held-out ${testPct}% Test Cohort (${testCount} Patients)`;

        // RENDER BASED ON NUMBER OF TESTED MODELS
        if (allResults.length > 1) {
            // MULTI-MODEL TOURNAMENT / COMPARISON MODE
            if (DOM.dashboardSingleHeading) DOM.dashboardSingleHeading.style.display = 'none';
            if (DOM.metricsSummaryRow) DOM.metricsSummaryRow.style.display = 'none';
            if (DOM.evaluationDetailContainer) DOM.evaluationDetailContainer.style.display = 'none';

            // 1. Benchmark Leaderboard Table
            if (DOM.multiModelLeaderboardContainer) {
                DOM.multiModelLeaderboardContainer.style.display = 'block';
                let lbRows = '';
                allResults.forEach((resItem, idx) => {
                    const mInfo = resItem.info;
                    const rEval = resItem.evalRes;
                    const isChampion = idx === 0;
                    const isDeployed = AppState.deployedModelId === resItem.modelId;

                    const rankBadge = isChampion ? '🥇 #1 Champion' : (idx === 1 ? '🥈 #2 Runner-Up' : (idx === 2 ? '🥉 #3' : `#${idx + 1}`));
                    const scoreDisplay = rEval.isClassification 
                        ? `<strong style="color: var(--accent-cyan); font-size: 14.5px;">${(rEval.accuracy * 100).toFixed(1)}%</strong>`
                        : `<strong style="color: var(--accent-cyan); font-size: 14.5px;">R²: ${rEval.r2.toFixed(3)}</strong>`;

                    const rocAucDisplay = rEval.isClassification 
                        ? (rEval.rocAuc !== null ? rEval.rocAuc.toFixed(3) : '<span style="color:var(--text-muted)">N/A</span>')
                        : `RMSE: ${rEval.rmse.toFixed(3)}`;

                    const f1Display = rEval.isClassification 
                        ? `${(rEval.macroF1 * 100).toFixed(1)}%`
                        : `MAE: ${rEval.mae.toFixed(3)}`;

                    const precDisplay = rEval.isClassification 
                        ? `${(rEval.macroPrecision * 100).toFixed(1)}%`
                        : `MSE: ${rEval.mse.toFixed(3)}`;

                    const recDisplay = rEval.isClassification 
                        ? `${(rEval.macroRecall * 100).toFixed(1)}%`
                        : '-';

                    const seedDisplay = (rEval.seed !== null && rEval.seed !== undefined)
                        ? `<span class="seed-pill" title="Click to copy reproducibility seed: ${rEval.seed}" onclick="window.copySeedToClipboard(${rEval.seed})">🌱 ${rEval.seed} <span class="copy-icon">📋</span></span>`
                        : `<span style="font-size: 11px; color: var(--text-muted); font-family: monospace;">N/A</span>`;

                    const deployBtn = isDeployed
                        ? `<span class="pill pill-target" style="background: rgba(16, 185, 129, 0.2); color: var(--accent-emerald); font-weight: 700; border-color: rgba(16, 185, 129, 0.4);">✓ Active Predictor</span>`
                        : `<button class="btn-preset-action deploy-model-btn" onclick="window.deployTestedModel('${resItem.modelId}')" style="padding: 4px 10px; font-size: 11px; cursor: pointer;">Deploy Model</button>`;

                    lbRows += `
                        <tr style="${isDeployed ? 'background: rgba(6, 182, 212, 0.05);' : ''}">
                            <td><span class="leaderboard-rank ${isChampion ? 'rank-1' : ''}">${rankBadge}</span></td>
                            <td>
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <span style="font-size: 16px;">${mInfo.icon}</span>
                                    <div>
                                        <div style="font-weight: 700; color: var(--text-primary);">${mInfo.name}</div>
                                        <div style="font-size: 11px; color: var(--text-muted);">${resItem.retrained ? '100% Cohort Retrained' : (isOptionB ? 'External Validated' : 'Split-Trained')}</div>
                                    </div>
                                </div>
                            </td>
                            <td><span class="pill ${mInfo.pillClass}" style="font-size: 11px;">${mInfo.tag}</span></td>
                            <td>${scoreDisplay}</td>
                            <td>${rocAucDisplay}</td>
                            <td>${f1Display}</td>
                            <td>${precDisplay}</td>
                            <td>${recDisplay}</td>
                            <td>${seedDisplay}</td>
                            <td>${deployBtn}</td>
                        </tr>
                    `;
                });

                DOM.multiModelLeaderboardContainer.innerHTML = `
                    <div style="background: var(--bg-surface-elevated); border: 1px solid var(--border-color); border-radius: var(--radius-lg); padding: 18px 20px; margin-bottom: 24px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 14px;">
                            <div>
                                <h3 style="font-size: 16px; font-weight: 700; color: var(--text-primary); display: flex; align-items: center; gap: 8px;">
                                    <span>🏆</span> Benchmark Leaderboard &mdash; ${allResults.length} Tested Models Ranked
                                </h3>
                                <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">
                                    Comparative diagnostic performance on ${cohortDesc}. Top-performing model (${allResults[0].info.name}) is auto-designated as active predictor. Click any seed pill to copy its reproducible reference.
                                </div>
                            </div>
                            <span class="pill pill-target" style="font-size: 12px; font-weight: 700;">
                                ${allResults.length} Models Tested
                            </span>
                        </div>
                        <div class="table-container">
                            <table class="leaderboard-table">
                                <thead>
                                    <tr>
                                        <th>Rank</th>
                                        <th>Model Algorithm</th>
                                        <th>Paradigm</th>
                                        <th>${prep.isClassification ? 'Test Accuracy' : 'Test R²'}</th>
                                        <th>${prep.isClassification ? 'ROC-AUC' : 'RMSE'}</th>
                                        <th>${prep.isClassification ? 'Macro F1' : 'MAE'}</th>
                                        <th>${prep.isClassification ? 'Precision' : 'MSE'}</th>
                                        <th>${prep.isClassification ? 'Recall' : '-'}</th>
                                        <th>Seed (Ref)</th>
                                        <th>Status / Deploy</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${lbRows}
                                </tbody>
                            </table>
                        </div>
                    </div>
                `;
            }

            // 2. Individual Confusion Matrices & Accuracy Cards for ALL Chosen Models
            if (DOM.multiModelCmContainer) {
                DOM.multiModelCmContainer.style.display = 'block';
                let cmCardsHtml = '';
                allResults.forEach((resItem, idx) => {
                    const mInfo = resItem.info;
                    const rEval = resItem.evalRes;
                    const isChampion = idx === 0;
                    const isDeployed = AppState.deployedModelId === resItem.modelId;

                    let cmContent = '';
                    if (rEval.isClassification) {
                        const cm = rEval.confMatrix;
                        const classes = prep.targetClasses;
                        cmContent = `
                            <div class="table-container" style="margin-top: 10px;">
                                <table class="cm-table" style="font-size: 11.5px;">
                                    <thead>
                                        <tr>
                                            <th style="padding: 6px 8px;">Actual \\ Pred</th>
                                            ${classes.map(c => `<th style="padding: 6px 8px;">${c}</th>`).join('')}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        ${classes.map((actualClass, r) => `
                                            <tr>
                                                <th style="text-align: left; padding: 6px 8px;">${actualClass}</th>
                                                ${classes.map((_, c) => {
                                                    const val = cm[r][c];
                                                    const isHit = r === c;
                                                    return `<td class="${isHit ? 'cm-cell-hit' : 'cm-cell-miss'}" style="padding: 6px 8px; font-weight: 700;">${val}</td>`;
                                                }).join('')}
                                            </tr>
                                        `).join('')}
                                    </tbody>
                                </table>
                            </div>
                        `;
                    } else {
                        const cm = rEval.confMatrix || [[0, 0], [0, 0]];
                        const thresh = rEval.threshold !== undefined ? rEval.threshold : 0;
                        const regClasses = [`< ${thresh}`, `≥ ${thresh}`];
                        const accPct = rEval.thresholdAccuracy !== undefined ? (rEval.thresholdAccuracy * 100).toFixed(1) : 'N/A';
                        const sensPct = rEval.sensitivity !== undefined ? (rEval.sensitivity * 100).toFixed(1) : 'N/A';
                        const specPct = rEval.specificity !== undefined ? (rEval.specificity * 100).toFixed(1) : 'N/A';
                        cmContent = `
                            <div class="table-container" style="margin-top: 10px;">
                                <div style="font-size: 11px; font-weight: 700; color: var(--text-secondary); margin-bottom: 5px; display: flex; justify-content: space-between; align-items: center;">
                                    <span>2×2 Concordance Matrix (Cutoff: ${thresh})</span>
                                    <span style="color: var(--accent-emerald); font-weight: 700;">${accPct}% Acc</span>
                                </div>
                                <table class="cm-table" style="font-size: 11.5px; width: 100%;">
                                    <thead>
                                        <tr>
                                            <th style="padding: 6px 8px; font-size: 10px;">Actual \\ Pred</th>
                                            ${regClasses.map(c => `<th style="padding: 6px 8px; font-size: 10.5px;">${c}</th>`).join('')}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        ${regClasses.map((actualClass, r) => `
                                            <tr>
                                                <th style="text-align: left; padding: 6px 8px; font-size: 10.5px;">${actualClass}</th>
                                                ${regClasses.map((_, c) => {
                                                    const val = cm[r][c];
                                                    const isHit = r === c;
                                                    return `<td class="${isHit ? 'cm-cell-hit' : 'cm-cell-miss'}" style="padding: 6px 8px; font-weight: 700;">${val}</td>`;
                                                }).join('')}
                                            </tr>
                                        `).join('')}
                                    </tbody>
                                </table>
                                <div style="display: flex; justify-content: space-between; margin-top: 6px; font-size: 11px; color: var(--text-secondary); padding: 0 4px;">
                                    <span>Sens: <strong>${sensPct}%</strong></span>
                                    <span>Spec: <strong>${specPct}%</strong></span>
                                    <span>MAE: <strong>${rEval.mae.toFixed(3)}</strong></span>
                                </div>
                                <div style="margin-top: 6px; padding: 6px 8px; background: var(--bg-surface-subtle); border-radius: var(--radius-sm); font-size: 10.5px; color: var(--text-muted); line-height: 1.35;">
                                    💡 <em>2×2 matrix dichotomized at median cutoff (${thresh}). For discrete binary categories, choose <strong>Nominal</strong> or <strong>Ordinal</strong> in Step 2.</em>
                                </div>
                            </div>
                        `;
                    }

                    const deployBtn = isDeployed
                        ? `<button class="btn-preset-action" disabled style="background: rgba(16, 185, 129, 0.15); border-color: rgba(16, 185, 129, 0.4); color: var(--accent-emerald); font-weight: 700; width: 100%;">✓ Active Predictor (Steps 5 &amp; 6)</button>`
                        : `<button class="btn-preset-action deploy-model-btn" onclick="window.deployTestedModel('${resItem.modelId}')" style="width: 100%; font-weight: 600; cursor: pointer;">⭐ Set as Active Predictor</button>`;

                    cmCardsHtml += `
                        <div class="multi-cm-card ${isDeployed ? 'active-deployed' : ''}">
                            <div class="multi-cm-card-header">
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <span style="font-size: 18px;">${mInfo.icon}</span>
                                    <div>
                                        <div class="cm-card-title">${mInfo.name}</div>
                                        <span class="pill ${mInfo.pillClass}" style="font-size: 10px; padding: 1px 6px;">${mInfo.tag}</span>
                                    </div>
                                </div>
                                <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 4px;">
                                    ${isChampion ? `<span class="champion-badge">🏆 #1 Champion</span>` : `<span class="pill" style="font-size: 10.5px;">Rank #${idx + 1}</span>`}
                                    ${isDeployed ? `<span class="pill pill-target" style="font-size: 10px; background: rgba(16, 185, 129, 0.15); color: var(--accent-emerald);">⭐ Active</span>` : ''}
                                </div>
                            </div>

                            <div class="cm-card-metrics">
                                <div class="cm-metric-item">
                                    <span class="cm-m-label">${rEval.isClassification ? 'Test Accuracy' : 'R² Score'}</span>
                                    <span class="cm-m-val">${rEval.isClassification ? `${(rEval.accuracy * 100).toFixed(1)}%` : rEval.r2.toFixed(3)}</span>
                                </div>
                                <div class="cm-metric-item">
                                    <span class="cm-m-label">${rEval.isClassification ? 'ROC-AUC' : 'RMSE'}</span>
                                    <span class="cm-m-val" style="color: var(--accent-purple);">${rEval.isClassification ? (rEval.rocAuc !== null ? rEval.rocAuc.toFixed(3) : 'N/A') : rEval.rmse.toFixed(3)}</span>
                                </div>
                                <div class="cm-metric-item">
                                    <span class="cm-m-label">${rEval.isClassification ? 'Macro F1' : 'MAE'}</span>
                                    <span class="cm-m-val" style="color: var(--accent-amber);">${rEval.isClassification ? `${(rEval.macroF1 * 100).toFixed(1)}%` : rEval.mae.toFixed(3)}</span>
                                </div>
                            </div>

                            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px; padding-top: 6px; border-top: 1px dashed var(--border-color);">
                                <div class="cm-seed-card-badge" title="Click to copy reproducibility seed: ${rEval.seed}" onclick="window.copySeedToClipboard(${rEval.seed})">
                                    <span>🌱 Seed:</span> <strong>${rEval.seed !== null && rEval.seed !== undefined ? rEval.seed : 'N/A'}</strong> <span style="font-size: 10px; opacity: 0.8;">📋</span>
                                </div>
                                ${rEval.splitSeed ? `<span style="font-size: 10.5px; color: var(--text-muted); font-family: monospace;">Split: ${rEval.splitSeed}</span>` : ''}
                            </div>

                            <div style="font-size: 12px; font-weight: 700; color: var(--text-primary); margin-top: 10px;">
                                ${rEval.isClassification ? 'Confusion Matrix:' : '2×2 Confusion Matrix &amp; Diagnostics:'}
                            </div>
                            ${cmContent}

                            <div style="margin-top: 12px;">
                                ${deployBtn}
                            </div>
                        </div>
                    `;
                });

                DOM.multiModelCmContainer.innerHTML = `
                    <div style="margin-bottom: 20px;">
                        <h3 style="font-size: 16px; font-weight: 700; color: var(--text-primary); display: flex; align-items: center; gap: 8px;">
                            <span>📊</span> Confusion Matrix &amp; Diagnostic Accuracy Breakdown &mdash; All Chosen Models
                        </h3>
                        <p style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">
                            Individual confusion matrices and test metrics for every model chosen in this benchmark. Each card includes its exact reproducibility seed (click to copy).
                        </p>
                    </div>
                    <div class="multi-cm-grid">
                        ${cmCardsHtml}
                    </div>
                `;
            }
        } else {
            // SINGLE MODEL TESTING MODE
            if (DOM.multiModelLeaderboardContainer) DOM.multiModelLeaderboardContainer.style.display = 'none';
            if (DOM.multiModelCmContainer) DOM.multiModelCmContainer.style.display = 'none';
            if (DOM.dashboardSingleHeading) DOM.dashboardSingleHeading.style.display = 'block';
            if (DOM.metricsSummaryRow) DOM.metricsSummaryRow.style.display = 'grid';
            if (DOM.evaluationDetailContainer) DOM.evaluationDetailContainer.style.display = 'block';

            const activeInfo = MODEL_INFO[AppState.deployedModelId || AppState.selectedModelId] || { name: 'Model', icon: '🧠' };
            if (DOM.dashboardSingleHeading) {
                DOM.dashboardSingleHeading.innerHTML = `
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span>${activeInfo.icon}</span> Single Model Diagnostic Performance: <strong>${activeInfo.name}</strong>
                    </div>
                `;
            }

            const testAccLabel = isOptionB ? 'External Test Accuracy (Dataset ii)' : `Test Accuracy (${testPct}% Split)`;
            const testR2Label = isOptionB ? 'External Test R² (Dataset ii)' : `Test R² Score (${testPct}% Split)`;

            if (evalRes.isClassification) {
                DOM.metricsSummaryRow.innerHTML = `
                    <div class="metric-pill-card">
                        <span class="m-title">${testAccLabel}</span>
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
                    <div class="metric-pill-card" style="cursor: pointer;" onclick="window.copySeedToClipboard(${evalRes.seed})" title="Click to copy reproducibility seed: ${evalRes.seed}">
                        <span class="m-title">Reproducibility Seed</span>
                        <span class="m-val" style="color: var(--accent-purple); font-family: monospace;">🌱 ${evalRes.seed !== null && evalRes.seed !== undefined ? evalRes.seed : 'Auto'} 📋</span>
                    </div>
                `;

                // Confusion Matrix Table
                const cm = evalRes.confMatrix;
                const classes = prep.targetClasses;
                let cmHtml = `
                    <div style="display: flex; flex-direction: column; gap: 16px;">
                        <h3 style="font-size: 15px; color: var(--text-primary);">Confusion Matrix (${cohortDesc})</h3>
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
                        <span class="m-title">${testR2Label}</span>
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
                        <span class="m-title">2×2 Concordance Acc</span>
                        <span class="m-val" style="color: var(--accent-emerald);">${evalRes.thresholdAccuracy !== undefined ? `${(evalRes.thresholdAccuracy * 100).toFixed(1)}%` : evalRes.mse.toFixed(3)}</span>
                    </div>
                    <div class="metric-pill-card" style="cursor: pointer;" onclick="window.copySeedToClipboard(${evalRes.seed})" title="Click to copy reproducibility seed: ${evalRes.seed}">
                        <span class="m-title">Reproducibility Seed</span>
                        <span class="m-val" style="color: var(--accent-purple); font-family: monospace;">🌱 ${evalRes.seed !== null && evalRes.seed !== undefined ? evalRes.seed : 'Auto'} 📋</span>
                    </div>
                `;

                const cm = evalRes.confMatrix;
                const thresh = evalRes.threshold !== undefined ? evalRes.threshold : 0;
                const regClasses = [`< ${thresh}`, `≥ ${thresh}`];
                let regCmHtml = '';
                if (cm) {
                    regCmHtml = `
                        <div style="display: flex; flex-direction: column; gap: 14px; margin-bottom: 16px;">
                            <div style="display: flex; justify-content: space-between; align-items: baseline; max-width: 500px;">
                                <h3 style="font-size: 15px; color: var(--text-primary); margin: 0;">2×2 Concordance Confusion Matrix (${cohortDesc})</h3>
                                <span class="pill pill-target" style="font-size: 11px;">Cutoff: ${thresh}</span>
                            </div>
                            <div class="table-container" style="max-width: 500px;">
                                <table class="cm-table">
                                    <thead>
                                        <tr>
                                            <th>Actual \\ Predicted</th>
                                            ${regClasses.map(c => `<th>${c}</th>`).join('')}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        ${regClasses.map((actualClass, r) => `
                                            <tr>
                                                <th style="text-align: left;">${actualClass}</th>
                                                ${regClasses.map((_, c) => {
                                                    const val = cm[r][c];
                                                    const isHit = r === c;
                                                    return `<td class="${isHit ? 'cm-cell-hit' : 'cm-cell-miss'}">${val}</td>`;
                                                }).join('')}
                                            </tr>
                                        `).join('')}
                                    </tbody>
                                </table>
                            </div>
                            <div style="max-width: 500px; display: flex; justify-content: space-between; font-size: 12px; color: var(--text-secondary); background: var(--bg-surface-subtle); padding: 8px 12px; border-radius: var(--radius-sm);">
                                <span>Sensitivity: <strong style="color: var(--text-primary);">${(evalRes.sensitivity * 100).toFixed(1)}%</strong></span>
                                <span>Specificity: <strong style="color: var(--text-primary);">${(evalRes.specificity * 100).toFixed(1)}%</strong></span>
                                <span>Concordance Acc: <strong style="color: var(--accent-emerald);">${(evalRes.thresholdAccuracy * 100).toFixed(1)}%</strong></span>
                            </div>
                        </div>
                    `;
                }

                DOM.evaluationDetailContainer.innerHTML = `
                    ${regCmHtml}
                    <p style="font-size: 13px; color: var(--text-secondary); margin-top: 4px;">
                        Model validated on ${cohortDesc} with R² = ${evalRes.r2} and Mean Absolute Error = ${evalRes.mae}. ${isOptionB ? `Fitted strictly on 100% of Training Dataset (i) (${trainCount} patients).` : (evalRes.retrainOn100 ? `Active model was then retrained on 100% of data (${totalCount} patients).` : `Active model maintained strictly on ${trainPct}% training split (${trainCount} patients; 100% retraining declined).`)}
                    </p>
                    <div style="font-size: 11.5px; color: var(--text-muted); margin-top: 6px;">
                        💡 <em>Note: For continuous regression targets, the 2×2 Confusion Matrix is generated by clinical dichotomization at the median threshold (${thresh}). To train as categorical classification, select <strong>Nominal</strong> or <strong>Ordinal</strong> in Step 2.</em>
                    </div>
                `;
            }
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

        const activeSeed = (AppState.trainedModel && AppState.trainedModel.seed !== undefined && AppState.trainedModel.seed !== null)
            ? AppState.trainedModel.seed
            : (AppState.evalResults?.seed !== null && AppState.evalResults?.seed !== undefined ? AppState.evalResults.seed : null);

        const seedSnippet = activeSeed !== null
            ? ` &bull; Reproducibility Seed: <code style="color: var(--accent-purple); font-weight: 700; background: rgba(168,85,247,0.12); padding: 1px 6px; border-radius: 4px; cursor: pointer;" onclick="window.copySeedToClipboard(${activeSeed})" title="Click to copy reproducibility seed">🌱 ${activeSeed} 📋</code>`
            : '';

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
                        Active Model: <strong>${MODEL_INFO[AppState.deployedModelId || AppState.selectedModelId]?.name || (AppState.deployedModelId || AppState.selectedModelId).toUpperCase()}</strong> &bull; Target: <em>${prep.targetCol}</em>${seedSnippet}
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
                        Active Model: <strong>${MODEL_INFO[AppState.deployedModelId || AppState.selectedModelId]?.name || (AppState.deployedModelId || AppState.selectedModelId).toUpperCase()}</strong> &bull; Unit/Scale: <em>${prep.targetCol}</em>${seedSnippet}
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
        const activeSeed = (AppState.trainedModel && AppState.trainedModel.seed !== undefined && AppState.trainedModel.seed !== null)
            ? AppState.trainedModel.seed
            : (AppState.evalResults?.seed !== null && AppState.evalResults?.seed !== undefined ? AppState.evalResults.seed : 'Auto');

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
            rowCopy[`Model_Seed`] = activeSeed;
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
        setModelScope('single');
    });

})();
