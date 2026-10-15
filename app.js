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
        deployedModelId: 'nn',
        // 7-Step Clinical ML Pipeline State
        pipeline: null // initialised below via createEmptyPipelineState()
    };

    AppState.pipeline = createEmptyPipelineState(42);

    // Model Metadata & Lookup
    const MODEL_INFO = {
        nn: { name: 'Neural Network (MLP)', short: 'Neural Network', tag: 'Neural Network', pillClass: 'pill-target', icon: '🧠' },
        rf: { name: 'Random Forest', short: 'Random Forest', tag: 'Ensemble Trees', pillClass: 'pill-feature', icon: '🌲' },
        glm: { name: 'Logistic / Ridge Regression', short: 'GLM / Regression', tag: 'Generalized Linear', pillClass: 'pill-continuous', icon: '📈' },
        knn: { name: 'k-Nearest Neighbors (k-NN)', short: 'k-NN', tag: 'Instance-Based', pillClass: 'pill-ordinal', icon: '🎯' },
        nb: { name: 'Naive Bayes (Gaussian + Bernoulli)', short: 'Naive Bayes', tag: 'Probabilistic', pillClass: 'pill-nominal', icon: '🎲' },
        ensemble: { name: 'Voting / Weighted Ensemble', short: 'Ensemble', tag: 'Voting Ensemble', pillClass: 'pill-target', icon: '⚡' }
    };
    const ALL_MODEL_IDS = ['nn', 'rf', 'glm', 'knn', 'nb', 'ensemble'];

    // Escapes text for safe insertion into HTML (column names and cell values come from user CSVs)
    function escapeHtml(value) {
        return String(value === undefined || value === null ? '' : value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    // Encodes a value as a JS literal for use inside an inline onclick="..." attribute
    function jsArg(value) {
        return escapeHtml(JSON.stringify(String(value)));
    }

    // Median of training outcomes: the exploratory 2×2 cutoff for continuous targets is fixed from
    // TRAINING data so the test outcomes never influence the threshold
    function trainingMedian(values) {
        const v = values.filter(x => typeof x === 'number' && isFinite(x)).sort((a, b) => a - b);
        if (v.length === 0) return null;
        const mid = Math.floor(v.length / 2);
        return v.length % 2 === 0 ? (v[mid - 1] + v[mid]) / 2 : v[mid];
    }

    // Exploratory 2×2 table for a continuous outcome. The cutoff is fixed from TRAINING data (never
    // the test outcomes); it is not a validated clinical decision threshold, so it is labelled as such.
    function renderExploratoryCutoffTable(ev, compact) {
        if (!ev || !ev.confMatrix || typeof ev.threshold !== 'number') return '';
        const cm = ev.confMatrix;
        const thresh = ev.threshold;
        const labels = [`< ${thresh}`, `≥ ${thresh}`];
        const pct = v => (typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : 'N/A');
        const pad = compact ? 'padding: 6px 8px; font-size: 10.5px;' : '';
        return `
            <div class="table-container" style="margin-top: 10px; ${compact ? '' : 'max-width: 500px;'}">
                <div style="font-size: ${compact ? 11 : 13}px; font-weight: 700; color: var(--text-secondary); margin-bottom: 5px;">
                    Exploratory split at ${thresh} (${escapeHtml(ev.thresholdSource || 'training median')})
                </div>
                <table class="cm-table" style="${compact ? 'font-size: 11.5px; width: 100%;' : ''}">
                    <thead>
                        <tr>
                            <th style="${pad}">Actual \\ Pred</th>
                            ${labels.map(c => `<th style="${pad}">${escapeHtml(c)}</th>`).join('')}
                        </tr>
                    </thead>
                    <tbody>
                        ${labels.map((actual, r) => `
                            <tr>
                                <th style="text-align: left; ${pad}">${escapeHtml(actual)}</th>
                                ${labels.map((_, c) => `<td class="${r === c ? 'cm-cell-hit' : 'cm-cell-miss'}" style="${pad} font-weight: 700;">${cm[r][c]}</td>`).join('')}
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
                <div style="display: flex; justify-content: space-between; gap: 8px; margin-top: 6px; font-size: 11px; color: var(--text-secondary); padding: 0 4px;">
                    <span>Agreement: <strong>${pct(ev.thresholdAccuracy)}</strong></span>
                    <span>"Sens" (≥ cutoff): <strong>${pct(ev.sensitivity)}</strong></span>
                    <span>"Spec" (&lt; cutoff): <strong>${pct(ev.specificity)}</strong></span>
                </div>
                <div style="margin-top: 6px; padding: 6px 8px; background: var(--bg-surface-subtle); border-radius: var(--radius-sm); font-size: 10.5px; color: var(--text-muted); line-height: 1.35;">
                    ⚠️ <em>Exploratory only: the cutoff is the training-data median, not a prespecified clinical threshold, so these are not diagnostic accuracy measures.</em>
                </div>
            </div>
        `;
    }

    // What a displayed class "probability" actually is, by model type
    const PROBABILITY_LABELS = {
        model: { title: 'Model Probability', note: 'Model-estimated class probability' },
        blend: { title: 'Averaged Probability', note: 'Average of the base models\' class probabilities/vote shares' },
        vote: { title: 'Vote Share', note: 'Proportion of trees/models voting for this class' },
        neighbours: { title: 'Neighbour Vote Share', note: 'Share of the k nearest training cases (distance-weighted if selected) in this class' }
    };

    function probabilityLabel(type) {
        return PROBABILITY_LABELS[type] || PROBABILITY_LABELS.model;
    }

    // Display name for a trained model, flagging a neural network that fell back to a linear model
    function displayModelName(modelId, model) {
        const base = (MODEL_INFO[modelId] && MODEL_INFO[modelId].name) || String(modelId).toUpperCase();
        return model && model.usedFallback ? `${base} ⚠️ linear fallback (not a neural network)` : base;
    }

    // Strict integer seed parsing: parseInt would read "1e6" as 1 and "42.5" as 42, silently using a
    // different seed from the one typed. Returns null for anything that is not a plain integer.
    function parseSeed(raw) {
        const t = String(raw === undefined || raw === null ? '' : raw).trim();
        return /^-?\d+$/.test(t) ? parseInt(t, 10) : null;
    }

    function fmtR2(v) {
        return typeof v === 'number' && isFinite(v) ? v.toFixed(3) : 'N/A';
    }

    // Plain-language summary of how a neural network was trained (validation split / early stopping)
    function describeNNTraining(info) {
        if (!info) return '';
        if (!info.earlyStopping) {
            return `trained on all ${info.fitCases} training cases for ${info.maxEpochs} epochs (too few cases for an internal validation split; no early stopping).`;
        }
        return `${info.validationCases} training cases held out for validation; ` +
            (info.earlyStopped
                ? `early stopping at epoch ${info.stoppedEpoch}/${info.maxEpochs}, weights restored from epoch ${info.bestEpoch}.`
                : `ran all ${info.maxEpochs} epochs, best weights from epoch ${info.bestEpoch}.`) +
            ' The test set was not used during training.';
    }

    function cssEscape(value) {
        return (window.CSS && typeof window.CSS.escape === 'function') ? window.CSS.escape(value) : String(value).replace(/["\\]/g, '\\$&');
    }

    // Ranking policy (one policy for Step 4 and the 7-step pipeline; selectable by the user so the
    // primary metric can match the prespecified study objective)
    const RANKING_METRICS = {
        classification: {
            macroF1: { label: 'Macro F1', higherIsBetter: true },
            balancedAccuracy: { label: 'Balanced accuracy', higherIsBetter: true },
            rocAuc: { label: 'ROC-AUC (binary only)', higherIsBetter: true },
            brierScore: { label: 'Brier score (lower is better)', higherIsBetter: false },
            accuracy: { label: 'Accuracy', higherIsBetter: true }
        },
        regression: {
            r2: { label: 'R²', higherIsBetter: true },
            rmse: { label: 'RMSE (lower is better)', higherIsBetter: false },
            mae: { label: 'MAE (lower is better)', higherIsBetter: false }
        }
    };

    function rankingMetricKey(isClassification) {
        return isClassification ? (AppState.rankingMetricClf || 'macroF1') : (AppState.rankingMetricReg || 'r2');
    }

    function rankingMetricLabel(isClassification) {
        const group = isClassification ? RANKING_METRICS.classification : RANKING_METRICS.regression;
        return group[rankingMetricKey(isClassification)].label;
    }

    // Sorts best-first on the selected metric; missing values rank last; ties broken by macro-F1/AUC
    // (classification) or R²/RMSE (regression).
    function compareEvalResults(a, b, isClassification) {
        const key = rankingMetricKey(isClassification);
        const def = (isClassification ? RANKING_METRICS.classification : RANKING_METRICS.regression)[key];
        const val = ev => (typeof ev[key] === 'number' && isFinite(ev[key]) ? ev[key] : null);
        const va = val(a);
        const vb = val(b);
        if (va === null && vb !== null) return 1;
        if (vb === null && va !== null) return -1;
        if (va !== null && vb !== null && va !== vb) return def.higherIsBetter ? vb - va : va - vb;
        if (isClassification) {
            return ((b.macroF1 || 0) - (a.macroF1 || 0)) || ((b.rocAuc || 0) - (a.rocAuc || 0)) || ((b.accuracy || 0) - (a.accuracy || 0));
        }
        // Regression fallback: higher R² first; if R² is equal or undefined for both (constant test
        // outcomes), fall back to lower RMSE rather than declaring a tie
        const r2a = typeof a.r2 === 'number' && isFinite(a.r2) ? a.r2 : null;
        const r2b = typeof b.r2 === 'number' && isFinite(b.r2) ? b.r2 : null;
        if (r2a !== null && r2b !== null && r2a !== r2b) return r2b - r2a;
        if (r2a !== null && r2b === null) return -1;
        if (r2a === null && r2b !== null) return 1;
        return (a.rmse || 0) - (b.rmse || 0);
    }

    function summariseHeldOutSet(ev) {
        // Caveats about the size/composition of the evaluation set
        const notes = [];
        // testCount is the held-out set size; totalSamples (from evaluate()) is the same number
        const n = ev.testCount || ev.totalSamples || 0;
        if (n > 0 && n < 30) {
            notes.push(`Only ${n} test cases: performance estimates are imprecise${Array.isArray(ev.accuracyCI) && ev.accuracyCI.length === 2 ? ` (accuracy 95% CI ${(ev.accuracyCI[0] * 100).toFixed(0)}–${(ev.accuracyCI[1] * 100).toFixed(0)}%)` : ''}. Treat differences between models with caution.`);
        }
        if (ev.classesMissingFromTest && ev.classesMissingFromTest.length > 0) {
            notes.push(`Class(es) absent from the test set: ${ev.classesMissingFromTest.map(escapeHtml).join(', ')}. Their recall/F1 cannot be evaluated and are counted as 0 in macro averages.`);
        }
        return notes;
    }


    // =========================================================================
    // EXECUTION WATCHDOG: LIVE MONITOR, LOOP DETECTOR & STALL SENTINEL
    // =========================================================================
    class ExecutionWatchdog {
        constructor(config) {
            this.container = config.container;
            this.badge = config.badge;
            this.stateText = config.stateText;
            this.timerText = config.timerText;
            this.loopText = config.loopText;
            this.loopStatus = config.loopStatus;
            this.taskName = config.taskName;
            this.stepCounter = config.stepCounter;
            this.progressBar = config.progressBar;
            this.threadStatus = config.threadStatus;
            this.heartbeatTick = config.heartbeatTick;
            this.stuckHint = config.stuckHint;
            this.cancelBtn = config.cancelBtn;

            this.startTime = 0;
            this.lastTickTime = 0;
            this.timerInterval = null;
            this.aborted = false;
            this.transitionHistory = [];
            this.loopCount = 0;
            this.onCancelCallback = null;

            if (this.cancelBtn) {
                this.cancelBtn.addEventListener('click', () => this.forceAbort());
            }
        }

        start(taskDescription, totalSteps = 1, onCancel = null) {
            this.aborted = false;
            this.onCancelCallback = onCancel;
            this.startTime = Date.now();
            this.lastTickTime = Date.now();
            this.totalSteps = totalSteps;
            this.currentStep = 1;
            this.transitionHistory = [];
            this.loopCount = 0;
            this.consecutiveRepeats = 0;

            if (this.container) this.container.style.display = 'block';
            if (this.badge) {
                this.badge.className = 'monitor-status-badge running';
            }
            if (this.stateText) this.stateText.innerHTML = 'ACTIVE &bull; COMPUTING';
            if (this.loopStatus) this.loopStatus.className = 'monitor-loop-status';
            if (this.loopText) this.loopText.innerText = 'Loop Guard: Normal (0 cycles)';
            if (this.stuckHint) {
                this.stuckHint.style.color = 'var(--accent-emerald)';
                this.stuckHint.innerText = 'System operating smoothly';
            }
            if (this.threadStatus) {
                this.threadStatus.style.color = 'var(--accent-emerald)';
                this.threadStatus.innerText = 'Responsive';
            }
            if (this.progressBar) {
                this.progressBar.style.width = '5%';
                this.progressBar.setAttribute('aria-valuenow', '5');
            }
            if (this.taskName) this.taskName.innerText = taskDescription;
            if (this.stepCounter) this.stepCounter.innerText = totalSteps > 1 ? `Step 1 of ${totalSteps}` : 'In progress...';

            // Start timer interval (updates every 100ms)
            if (this.timerInterval) clearInterval(this.timerInterval);
            this.timerInterval = setInterval(() => this.onTimerTick(), 100);
        }

        onTimerTick() {
            if (this.aborted) return;
            const now = Date.now();
            const elapsed = (now - this.startTime) / 1000;
            const mins = Math.floor(elapsed / 60);
            const secs = Math.floor(elapsed % 60);
            const tenths = Math.floor((elapsed * 10) % 10);
            const timerStr = `⏱️ ${mins < 10 ? '0' + mins : mins}:${secs < 10 ? '0' + secs : secs}.${tenths}s`;

            if (this.timerText) this.timerText.innerText = timerStr;

            // Heartbeat stall check
            const gap = (now - this.lastTickTime) / 1000;
            if (this.heartbeatTick) {
                this.heartbeatTick.innerText = gap < 1 ? '< 1s ago' : `${gap.toFixed(1)}s ago`;
            }

            if (gap > 12) {
                // Potential stall warning
                if (this.badge) this.badge.className = 'monitor-status-badge stalled';
                if (this.stateText) this.stateText.innerText = '⚠️ HEAVY LOAD / POSSIBLE STALL';
                if (this.threadStatus) {
                    this.threadStatus.style.color = 'var(--accent-rose)';
                    this.threadStatus.innerText = 'High Latency';
                }
                if (this.stuckHint) {
                    this.stuckHint.style.color = 'var(--accent-rose)';
                    this.stuckHint.innerText = 'Calculations taking longer than expected. Force Stop takes effect at the end of the current epoch (neural network), tree (Random Forest) or 10 iterations (logistic/ridge); k-NN and Naive Bayes stop between models. A single tree or k-NN scan cannot be interrupted mid-computation.';
                }
            } else if (gap > 5) {
                // Elevated compute
                if (this.badge) this.badge.className = 'monitor-status-badge warning';
                if (this.stateText) this.stateText.innerText = '⚙️ BUSY COMPUTING';
                if (this.threadStatus) {
                    this.threadStatus.style.color = 'var(--accent-amber)';
                    this.threadStatus.innerText = 'Heavy Math';
                }
                if (this.stuckHint) {
                    this.stuckHint.style.color = 'var(--accent-amber)';
                    this.stuckHint.innerText = 'Complex gradient descent / tree construction in progress...';
                }
            } else {
                if (this.badge && this.badge.className.includes('warning')) {
                    this.badge.className = 'monitor-status-badge running';
                    if (this.stateText) this.stateText.innerHTML = 'ACTIVE &bull; COMPUTING';
                    if (this.threadStatus) {
                        this.threadStatus.style.color = 'var(--accent-emerald)';
                        this.threadStatus.innerText = 'Responsive';
                    }
                    if (this.stuckHint) {
                        this.stuckHint.style.color = 'var(--accent-emerald)';
                        this.stuckHint.innerText = 'System operating smoothly';
                    }
                }
            }
        }

        tick(stepIndex, taskDescription, details = '') {
            if (this.aborted) throw new Error("Operation halted by user.");
            this.lastTickTime = Date.now();
            this.currentStep = stepIndex;

            // 1. Repetition monitor (advisory only). Identical progress messages are normal during
            // multi-epoch training, so repetition is reported but never used to abort a run. Every
            // training loop in this app has a fixed iteration limit; genuine hangs are surfaced by the
            // heartbeat stall indicator instead.
            const transitionKey = `${stepIndex}:${taskDescription}`;
            const last = this.transitionHistory[this.transitionHistory.length - 1];
            this.consecutiveRepeats = (last && last.key === transitionKey) ? (this.consecutiveRepeats || 0) + 1 : 0;
            this.transitionHistory.push({ key: transitionKey, time: this.lastTickTime });
            if (this.transitionHistory.length > 200) this.transitionHistory.shift();

            if (this.consecutiveRepeats >= 50) {
                this.loopCount++;
                if (this.loopStatus) this.loopStatus.className = 'monitor-loop-status loop-detected';
                if (this.loopText) this.loopText.innerText = `⚠️ Progress message unchanged for ${this.consecutiveRepeats} updates (still running)`;
            } else {
                if (this.loopStatus) this.loopStatus.className = 'monitor-loop-status';
                if (this.loopText) this.loopText.innerText = 'Loop Guard: Normal (bounded iterations)';
            }

            // 2. Update UI
            if (this.taskName) this.taskName.innerText = taskDescription;
            if (this.stepCounter && this.totalSteps > 1) {
                this.stepCounter.innerText = `Step ${stepIndex} of ${this.totalSteps}`;
            }
            if (this.progressBar) {
                const pct = Math.min(100, Math.round((stepIndex / this.totalSteps) * 100));
                this.progressBar.style.width = `${pct}%`;
                this.progressBar.setAttribute('aria-valuenow', String(pct));
            }
        }

        finish(successMessage = 'Completed successfully') {
            if (this.timerInterval) clearInterval(this.timerInterval);
            this.timerInterval = null;

            if (this.badge) this.badge.className = 'monitor-status-badge completed';
            if (this.stateText) this.stateText.innerText = '✓ COMPLETED';
            if (this.progressBar) {
                this.progressBar.style.width = '100%';
                this.progressBar.setAttribute('aria-valuenow', '100');
            }
            if (this.taskName) this.taskName.innerText = successMessage;
            if (this.stuckHint) {
                this.stuckHint.style.color = 'var(--accent-cyan)';
                this.stuckHint.innerText = 'Results rendered successfully';
            }

            setTimeout(() => {
                if (this.container && !this.aborted) {
                    this.container.style.display = 'none';
                }
            }, 800);
        }

        forceAbort(reason = "Operation aborted by user.") {
            this.aborted = true;
            if (this.timerInterval) clearInterval(this.timerInterval);
            this.timerInterval = null;

            if (this.badge) this.badge.className = 'monitor-status-badge stalled';
            if (this.stateText) this.stateText.innerText = '🛑 ABORTED';
            if (this.taskName) this.taskName.innerText = reason;
            if (this.stuckHint) {
                this.stuckHint.style.color = 'var(--accent-rose)';
                this.stuckHint.innerText = 'Cancellation requested: remaining work is skipped at the next checkpoint.';
            }

            if (typeof this.onCancelCallback === 'function') {
                try {
                    this.onCancelCallback(reason);
                } catch (e) {
                    console.error("Cancel callback error:", e);
                }
            }

            setTimeout(() => {
                if (this.container) this.container.style.display = 'none';
            }, 2500);
        }
    }

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

        // 7-Step Clinical ML Pipeline DOM References
        DOM.tabPipeline = document.getElementById('tab-step-pipeline');
        DOM.pipelineDatasetName = document.getElementById('pipeline-dataset-name');
        DOM.pipelineLoadCardioBtn = document.getElementById('pipeline-load-cardio-btn');
        DOM.pipelineLoadShuntBtn = document.getElementById('pipeline-load-shunt-btn');
        DOM.pipelineSubPills = document.querySelectorAll('.pipeline-step-pill');
        DOM.pipelinePanels = document.querySelectorAll('.pipeline-step-panel');
        DOM.pipelineSeedInput = document.getElementById('pipeline-seed-input');
        DOM.pipelineSeedRandomBtn = document.getElementById('pipeline-seed-random-btn');
        DOM.pipelineImputeStrategy = document.getElementById('pipeline-impute-strategy');
        DOM.pipelineStep1RunBtn = document.getElementById('pipeline-step1-run-btn');
        DOM.pipelineStep1Results = document.getElementById('pipeline-step1-results');
        DOM.pipelineTrainCount = document.getElementById('pipeline-train-count');
        DOM.pipelineTestCount = document.getElementById('pipeline-test-count');
        DOM.pipelineMissingImputedCount = document.getElementById('pipeline-missing-imputed-count');
        DOM.pipelineImputerTableContainer = document.getElementById('pipeline-imputer-table-container');
        DOM.pipelineToStep2Btn = document.getElementById('pipeline-to-step2-btn');
        DOM.pipelineNormalityTableContainer = document.getElementById('pipeline-normality-table-container');
        DOM.pipelineCorrPairsContainer = document.getElementById('pipeline-corr-pairs-container');
        DOM.pipelineCorrPairCounter = document.getElementById('pipeline-corr-pair-counter');
        DOM.pipelineRunCorrBtn = document.getElementById('pipeline-run-corr-btn');
        DOM.pipelineToStep3Btn = document.getElementById('pipeline-to-step3-btn');
        DOM.pipelineVifTableContainer = document.getElementById('pipeline-vif-table-container');
        DOM.pipelineDropVifBtn = document.getElementById('pipeline-drop-vif-btn');
        DOM.pipelineRunVifBtn = document.getElementById('pipeline-run-vif-btn');
        DOM.pipelineToStep4Btn = document.getElementById('pipeline-to-step4-btn');
        DOM.pipelineUnivariateTableContainer = document.getElementById('pipeline-univariate-table-container');
        DOM.pipelineFilterSignificantBtn = document.getElementById('pipeline-filter-significant-btn');
        DOM.pipelineSelectAllUnivBtn = document.getElementById('pipeline-select-all-univ-btn');
        DOM.pipelineDropUnselectedUnivBtn = document.getElementById('pipeline-drop-unselected-univ-btn');
        DOM.pipelineUnivSelectedSummary = document.getElementById('pipeline-univ-selected-summary');
        DOM.pipelineToStep5Btn = document.getElementById('pipeline-to-step5-btn');
        DOM.pipelineSmotencContainer = document.getElementById('pipeline-smotenc-container');
        DOM.pipelineToStep6Btn = document.getElementById('pipeline-to-step6-btn');
        DOM.pipelineRunRfScoutBtn = document.getElementById('pipeline-run-rf-scout-btn');
        DOM.pipelineRfImportanceContainer = document.getElementById('pipeline-rf-importance-container');
        DOM.pipelineToStep7Btn = document.getElementById('pipeline-to-step7-btn');
        DOM.pipelineEvalSeedInput = document.getElementById('pipeline-eval-seed-input');
        DOM.pipelineTrainModelsBtn = document.getElementById('pipeline-train-models-btn');
        DOM.pipelineTrainStatus = document.getElementById('pipeline-train-status');
        DOM.pipelineTrainStatusText = document.getElementById('pipeline-train-status-text');
        DOM.pipelineResultsContainer = document.getElementById('pipeline-results-container');
        DOM.pipelineLeaderboardContainer = document.getElementById('pipeline-leaderboard-container');
        DOM.pipelineCmGrid = document.getElementById('pipeline-cm-grid');
        DOM.pipelineDownloadReportBtn = document.getElementById('pipeline-download-report-btn');
        DOM.pipelineDirectLaunchBtn = document.getElementById('pipeline-direct-launch-btn');
        DOM.singlePredictProvenanceBar = document.getElementById('single-predict-provenance-bar');
        DOM.batchPredictProvenanceBar = document.getElementById('batch-predict-provenance-bar');
        DOM.pipelineActiveDeploymentStatus = document.getElementById('pipeline-active-deployment-status');
        DOM.pipelineToSinglePredictBtn = document.getElementById('pipeline-to-single-predict-btn');
        DOM.pipelineToBatchPredictBtn = document.getElementById('pipeline-to-batch-predict-btn');
        DOM.pipelineCompareBaselineBtn = document.getElementById('pipeline-compare-baseline-btn');
        DOM.pipelineVsBaselinePanel = document.getElementById('pipeline-vs-baseline-panel');
        DOM.pipelineVsBaselineContent = document.getElementById('pipeline-vs-baseline-content');

        // Step 7 Pipeline Watchdog
        DOM.pipelineMonitor = {
            container: document.getElementById('pipeline-train-status'),
            badge: document.getElementById('pipeline-monitor-badge'),
            stateText: document.getElementById('pipeline-monitor-state'),
            timerText: document.getElementById('pipeline-monitor-timer'),
            loopStatus: document.getElementById('pipeline-monitor-loop-status'),
            loopText: document.getElementById('pipeline-monitor-loop-text'),
            taskName: document.getElementById('pipeline-train-status-text'),
            stepCounter: document.getElementById('pipeline-monitor-step-counter'),
            progressBar: document.getElementById('pipeline-monitor-progress-bar'),
            threadStatus: document.getElementById('pipeline-monitor-thread-status'),
            heartbeatTick: document.getElementById('pipeline-monitor-heartbeat-tick'),
            stuckHint: document.getElementById('pipeline-monitor-stuck-hint'),
            cancelBtn: document.getElementById('pipeline-monitor-cancel-btn')
        };

        // Step 4 Model Training Watchdog
        DOM.step4Monitor = {
            container: document.getElementById('step4-train-monitor'),
            badge: document.getElementById('step4-monitor-badge'),
            stateText: document.getElementById('step4-monitor-state'),
            timerText: document.getElementById('step4-monitor-timer'),
            loopStatus: document.getElementById('step4-monitor-loop-status'),
            loopText: document.getElementById('step4-monitor-loop-text'),
            taskName: document.getElementById('step4-monitor-task-name'),
            stepCounter: document.getElementById('step4-monitor-step-counter'),
            progressBar: document.getElementById('step4-monitor-progress-bar'),
            threadStatus: document.getElementById('step4-monitor-thread-status'),
            heartbeatTick: document.getElementById('step4-monitor-heartbeat-tick'),
            stuckHint: document.getElementById('step4-monitor-stuck-hint'),
            cancelBtn: document.getElementById('step4-monitor-cancel-btn')
        };

        AppState.watchdogs = {
            pipeline: new ExecutionWatchdog(DOM.pipelineMonitor),
            step4: new ExecutionWatchdog(DOM.step4Monitor)
        };
    }

    function setupEventListeners() {
        // Theme toggle
        DOM.themeToggle.addEventListener('click', toggleTheme);

        // Stepper Navigation
        DOM.stepTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const step = tab.dataset.step;
                goToStep(step);
            });
        });

        // 7-Step Pipeline Sub-stepper Nav
        if (DOM.pipelineSubPills) {
            DOM.pipelineSubPills.forEach(pill => {
                pill.addEventListener('click', () => {
                    const substep = parseInt(pill.dataset.substep, 10);
                    goToPipelineSubstep(substep);
                });
            });
        }

        // Pipeline Sample Cohort Quick Loaders
        if (DOM.pipelineLoadCardioBtn) {
            DOM.pipelineLoadCardioBtn.addEventListener('click', async () => {
                await loadSampleDataset('universal_medical_cohort.csv', 'Cardiometabolic Risk Cohort (Binary)');
                initPipelineUI();
            });
        }
        if (DOM.pipelineLoadShuntBtn) {
            DOM.pipelineLoadShuntBtn.addEventListener('click', async () => {
                await loadSampleDataset('medical_shunt_cohort.csv', 'Neurosurgical Shunt Response Cohort');
                initPipelineUI();
            });
        }

        if (DOM.pipelineSeedRandomBtn) {
            DOM.pipelineSeedRandomBtn.addEventListener('click', () => {
                const r = Math.floor(Math.random() * 90000) + 1000;
                DOM.pipelineSeedInput.value = r;
                if (DOM.pipelineEvalSeedInput) DOM.pipelineEvalSeedInput.value = r;
            });
        }

        // Pipeline Step Action Buttons
        if (DOM.pipelineStep1RunBtn) DOM.pipelineStep1RunBtn.addEventListener('click', executePipelineStep1);
        if (DOM.pipelineToStep2Btn) DOM.pipelineToStep2Btn.addEventListener('click', () => goToPipelineSubstep(2));
        if (DOM.pipelineRunCorrBtn) DOM.pipelineRunCorrBtn.addEventListener('click', renderPipelineStep2);
        if (DOM.pipelineToStep3Btn) DOM.pipelineToStep3Btn.addEventListener('click', () => goToPipelineSubstep(3));
        if (DOM.pipelineRunVifBtn) DOM.pipelineRunVifBtn.addEventListener('click', renderPipelineStep3);
        if (DOM.pipelineDropVifBtn) DOM.pipelineDropVifBtn.addEventListener('click', pipelineDropSelectedVIF);
        if (DOM.pipelineToStep4Btn) DOM.pipelineToStep4Btn.addEventListener('click', () => goToPipelineSubstep(4));
        if (DOM.pipelineFilterSignificantBtn) DOM.pipelineFilterSignificantBtn.addEventListener('click', filterSignificantUnivariateFeatures);
        if (DOM.pipelineSelectAllUnivBtn) DOM.pipelineSelectAllUnivBtn.addEventListener('click', selectAllUnivariateFeatures);
        if (DOM.pipelineDropUnselectedUnivBtn) DOM.pipelineDropUnselectedUnivBtn.addEventListener('click', dropUnselectedUnivariateFeatures);
        if (DOM.pipelineToStep5Btn) DOM.pipelineToStep5Btn.addEventListener('click', () => goToPipelineSubstep(5));
        if (DOM.pipelineToStep6Btn) DOM.pipelineToStep6Btn.addEventListener('click', () => goToPipelineSubstep(6));
        if (DOM.pipelineRunRfScoutBtn) DOM.pipelineRunRfScoutBtn.addEventListener('click', executePipelineStep6);
        if (DOM.pipelineToStep7Btn) DOM.pipelineToStep7Btn.addEventListener('click', () => goToPipelineSubstep(7));
        if (DOM.pipelineTrainModelsBtn) DOM.pipelineTrainModelsBtn.addEventListener('click', executePipelineStep7);
        if (DOM.pipelineDownloadReportBtn) DOM.pipelineDownloadReportBtn.addEventListener('click', downloadPipelineProtocolReport);
        if (DOM.pipelineDirectLaunchBtn) {
            DOM.pipelineDirectLaunchBtn.addEventListener('click', (e) => window.launch7StepPipeline(e));
        }
        if (DOM.pipelineToSinglePredictBtn) {
            DOM.pipelineToSinglePredictBtn.addEventListener('click', () => {
                if (!AppState.trainedModel || AppState.deployedSource !== '7step') {
                    autoDeployBestPipelineModel();
                }
                goToStep(5);
            });
        }
        if (DOM.pipelineToBatchPredictBtn) {
            DOM.pipelineToBatchPredictBtn.addEventListener('click', () => {
                if (!AppState.trainedModel || AppState.deployedSource !== '7step') {
                    autoDeployBestPipelineModel();
                }
                goToStep(6);
            });
        }
        if (DOM.pipelineCompareBaselineBtn) {
            DOM.pipelineCompareBaselineBtn.addEventListener('click', togglePipelineBaselineComparison);
        }

        // Next Step Buttons
        DOM.toStep2Btn.addEventListener('click', () => {
            if (!AppState.parsedData || !AppState.parsedData.rows || AppState.parsedData.rows.length === 0) {
                alert("Please select a sample medical cohort or upload a clinical CSV first.");
                return;
            }
            goToStep(2);
        });
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

        // Primary ranking metric (shared by Step 4 and the 7-step pipeline)
        document.querySelectorAll('.ranking-metric-select').forEach(sel => {
            sel.addEventListener('change', (e) => {
                const [kind, key] = e.target.value.split(':');
                if (kind === 'clf') AppState.rankingMetricClf = key;
                else AppState.rankingMetricReg = key;
                // Keep both selectors in sync (each <select> shows one selected option)
                document.querySelectorAll('.ranking-metric-select').forEach(other => { other.value = e.target.value; });
                rerankExistingResults();
            });
        });

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

        // Initial model-scope sync (setModelScope is local to this function)
        setModelScope('single');
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
            // Choose the validation file from the built-in cohort actually loaded, not from the file name
            const sample = AppState.sampleCohortFile ? SAMPLE_COHORTS[AppState.sampleCohortFile] : null;
            if (!sample) {
                alert("The built-in validation cohort only matches the built-in sample datasets. For your own data, please upload its separate testing CSV.");
                return;
            }
            const filename = sample.testFile;
            const label = sample.testLabel;

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
            // Step 2 types are 'continuous' | 'ordinal' | 'nominal'; only a continuous target is regression
            const isClf = targetCfg ? targetCfg.type !== 'continuous' : true;
            DOM.optBTrainSummary.innerHTML = `
                <div><strong>Source:</strong> ${escapeHtml(AppState.fileName || 'Active Dataset')}</div>
                <div><strong>Training Cases:</strong> ${AppState.imputedData.imputedRows.length} patients (100% used for model parameter fitting)</div>
                <div><strong>Target:</strong> <span style="color: var(--accent-cyan); font-weight: 700;">${escapeHtml(targetCol || 'Unspecified')}</span> (${isClf ? 'Classification' : 'Continuous Regression'})</div>
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
                    // Parsed rows carry every header as a key, so the first row's keys are the file's header
                    const testHeaders = Object.keys(AppState.separateTestRawRows[0] || {});
                    const hasTarget = testHeaders.includes(targetCol);
                    targetStatus = hasTarget 
                        ? `<span style="color: var(--accent-emerald); font-weight: 600;">✓ Target '${escapeHtml(targetCol)}' found</span>`
                        : `<span style="color: var(--accent-amber); font-weight: 600;">⚠️ Target '${escapeHtml(targetCol)}' not in header!</span>`;
                    const predictors = Object.keys(AppState.columnConfigs).filter(c => AppState.columnConfigs[c].role === 'feature');
                    const missingPredictors = predictors.filter(c => !testHeaders.includes(c));
                    if (missingPredictors.length > 0) {
                        targetStatus += `<br><span style="color: var(--accent-amber); font-weight: 600;">⚠️ Predictor column(s) missing from the testing file (would be filled from training statistics for every row): ${escapeHtml(missingPredictors.join(', '))}</span>`;
                    }
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
                        <div><strong>Testing File:</strong> ${escapeHtml(AppState.separateTestFileName || 'Testing Cohort CSV')}</div>
                        <div><strong>Total Test Cases:</strong> ${testValidCount} external patients</div>
                        <div><strong>Target Verification:</strong> ${targetStatus}</div>
                        <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">
                            🔒 Test data are encoded and scaled using training-cohort parameters only. Rows with a missing outcome are excluded; missing predictor values are filled from training statistics.
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
        // Re-apply the theme palette to the existing chart (colours are set per option, not via CSS)
        if (AppState.charts.loss) {
            const isDark = AppState.theme === 'dark';
            const grid = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)';
            const tick = isDark ? '#94a3b8' : '#64748b';
            const o = AppState.charts.loss.options;
            ['x', 'y'].forEach(ax => {
                o.scales[ax].grid.color = grid;
                o.scales[ax].ticks.color = tick;
                o.scales[ax].title.color = tick;
            });
            o.plugins.legend.labels.color = isDark ? '#f8fafc' : '#0f172a';
            if (o.plugins.title) o.plugins.title.color = tick;
            AppState.charts.loss.update('none');
        }
    }

    window.launch7StepPipeline = async function(event) {
        if (event) {
            try {
                if (typeof event.preventDefault === 'function') event.preventDefault();
                if (typeof event.stopPropagation === 'function') event.stopPropagation();
            } catch (err) {}
        }

        // If dataset is not loaded yet, auto-load the default clinical cohort so the user is never blocked
        if (!AppState.parsedData || !AppState.parsedData.rows || AppState.parsedData.rows.length === 0) {
            try {
                await loadSampleDataset('medical_shunt_cohort.csv', 'Neurosurgical Shunt Response Cohort');
            } catch (err) {
                console.warn("Auto-load shunt cohort error:", err);
            }
        }

        goToStep('pipeline');
        initPipelineUI();
        goToPipelineSubstep(AppState.pipeline && AppState.pipeline.substep ? AppState.pipeline.substep : 1);
    };

    function goToStep(stepNum) {
        // Enforce prerequisites
        if (stepNum !== 'pipeline') {
            const num = parseInt(stepNum, 10);
            if (num > 1 && !AppState.parsedData) {
                alert("Please upload a CSV file or load a sample dataset first.");
                return;
            }
            // Step 4 trains from Step 3's imputed data; Steps 5-6 only need a deployed model (which may
            // come from the 7-step pipeline without the standard imputation step having been run)
            if (num === 4 && !AppState.imputedData) {
                alert("Please complete the Data Imputation step first.");
                return;
            }
            if (num > 4 && !AppState.trainedModel) {
                alert("Please train a machine learning model first in Step 4.");
                return;
            }
        }

        AppState.currentStep = stepNum;

        DOM.stepTabs.forEach(tab => {
            const s = tab.dataset.step;
            const isActive = String(s) === String(stepNum);
            tab.classList.toggle('active', isActive);
            if (isActive) tab.setAttribute('aria-current', 'step'); else tab.removeAttribute('aria-current');
            const num = parseInt(s, 10);
            const targetNum = parseInt(stepNum, 10);
            if (!isNaN(num) && !isNaN(targetNum) && num < targetNum) {
                tab.classList.add('completed');
            }
        });

        DOM.stepContents.forEach(content => {
            const s = content.dataset.step;
            const isActive = String(s) === String(stepNum);
            content.classList.toggle('active', isActive);
            content.style.display = isActive ? 'flex' : 'none';
        });

        window.scrollTo({ top: 0, behavior: 'smooth' });

        if (stepNum === 2 || stepNum === '2') renderVariableConfigGrid();
        if (stepNum === 3 || stepNum === '3') renderImputationTable();
        if (stepNum === 4 || stepNum === '4') refreshOptionBSummaries();
        if (stepNum === 5 || stepNum === '5') renderSinglePredictionForm();
        if (stepNum === 'pipeline') {
            initPipelineUI();
            goToPipelineSubstep(AppState.pipeline && AppState.pipeline.substep ? AppState.pipeline.substep : 1);
        }
    }
    window.goToStep = goToStep;

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

    // Known built-in cohorts and their matching validation (testing) files
    const SAMPLE_COHORTS = {
        'universal_medical_cohort.csv': { id: 'cardio', testFile: 'universal_medical_test_cohort.csv', testLabel: 'Cardiometabolic Risk Test Cohort' },
        'medical_shunt_cohort.csv': { id: 'shunt', testFile: 'medical_batch_unseen_cases.csv', testLabel: 'Neurosurgical External Validation Cohort' },
        'medical_recovery_score.csv': { id: 'recovery', testFile: 'medical_recovery_test_cohort.csv', testLabel: 'Functional Recovery Test Cohort' }
    };

    async function loadSampleDataset(filename, displayName) {
        try {
            const resp = await fetch(filename);
            if (!resp.ok) throw new Error("Could not load sample dataset file.");
            const text = await resp.text();
            processCSVContent(text, displayName);
            // Recorded after processCSVContent (which resets it): identifies the built-in cohort explicitly
            AppState.sampleCohortFile = SAMPLE_COHORTS[filename] ? filename : null;
        } catch (err) {
            alert(`Error loading sample dataset: ${err.message}`);
        }
    }

    function createEmptyPipelineState(seed = 42) {
        return {
            substep: 1,
            seed,
            evalSeed: seed,
            imputeStrategy: 'median',
            trainRows: null,
            testRows: null,
            balancedTrainRows: null,
            imputer: null,
            featureCols: [],
            targetCol: null,
            columnConfigs: {},
            droppedFeatures: new Set(),
            normalityMap: {},
            corrPairs: [],
            vifResults: [],
            univariateResults: [],
            smotencResult: null,
            rfScout: null,
            rfImportance: [],
            top70Features: [],
            bottom30Features: [],
            benchmarkResults: [],
            excludedMissingTarget: 0
        };
    }

    // Clears every result derived from a previous dataset so nothing stale can be trained or predicted on
    function resetDerivedState() {
        // Free GPU memory of every model from the previous dataset (including the deployed one)
        const oldModels = [
            ...((AppState.allTestedResults || []).map(r => r.model)),
            ...(((AppState.pipeline && AppState.pipeline.benchmarkResults) || []).map(b => b.model)),
            AppState.trainedModel
        ];
        AppState.trainedModel = null;
        disposeModels(oldModels);
        AppState.imputedData = null;
        AppState.prepMeta = null;
        AppState.trainTestData = null;
        AppState.trainedModel = null;
        AppState.evalResults = null;
        AppState.allTestedResults = null;
        AppState.separateTestData = null;
        AppState.batchPredictions = null;
        AppState.deployedSource = null;
        AppState.deployedPipelineModel = null;
        AppState.baselinePrepMeta = null;
        if (AppState.charts.loss) {
            AppState.charts.loss.destroy();
            AppState.charts.loss = null;
        }
        AppState.pipeline = createEmptyPipelineState(AppState.pipeline ? AppState.pipeline.seed : 42);

        if (DOM.imputeAuditBox) DOM.imputeAuditBox.innerHTML = '';
        if (DOM.toStep4Btn) DOM.toStep4Btn.disabled = true;
        if (DOM.toStep5Btn) DOM.toStep5Btn.disabled = true;
        if (DOM.toStep6Btn) DOM.toStep6Btn.disabled = true;
        if (DOM.resultsDashboard) DOM.resultsDashboard.style.display = 'none';
        if (DOM.trainStatus) DOM.trainStatus.innerHTML = '';
        if (DOM.singleResultBanner) DOM.singleResultBanner.style.display = 'none';
        if (DOM.predictFormGrid) DOM.predictFormGrid.innerHTML = '';
        if (DOM.batchResultsCard) DOM.batchResultsCard.style.display = 'none';
        if (DOM.downloadBatchCsvBtn) DOM.downloadBatchCsvBtn.disabled = true;
        if (DOM.pipelineStep1Results) DOM.pipelineStep1Results.style.display = 'none';
        if (DOM.pipelineResultsContainer) DOM.pipelineResultsContainer.style.display = 'none';
        if (DOM.pipelineVsBaselinePanel) DOM.pipelineVsBaselinePanel.style.display = 'none';
        if (DOM.pipelineRfImportanceContainer) DOM.pipelineRfImportanceContainer.innerHTML = '';
        if (DOM.pipelineSmotencContainer) DOM.pipelineSmotencContainer.innerHTML = '';
        if (DOM.pipelineToStep7Btn) DOM.pipelineToStep7Btn.disabled = true;
        if (DOM.pipelineSubPills) DOM.pipelineSubPills.forEach(p => p.classList.remove('completed'));
        if (DOM.singlePredictProvenanceBar) DOM.singlePredictProvenanceBar.style.display = 'none';
        if (DOM.batchPredictProvenanceBar) DOM.batchPredictProvenanceBar.style.display = 'none';
        if (DOM.pipelineActiveDeploymentStatus) {
            DOM.pipelineActiveDeploymentStatus.innerHTML = 'Select any model from the leaderboard to deploy for real-time patient predictions.';
        }
        if (DOM.stepTabs) DOM.stepTabs.forEach(t => t.classList.remove('completed'));
    }

    function processCSVContent(csvText, name) {
        try {
            const parsed = DataEngine.parseCSV(csvText);
            resetDerivedState();
            AppState.sampleCohortFile = null; // set again by loadSampleDataset for built-in cohorts
            AppState.parsedData = parsed;
            AppState.fileName = name;
            if (parsed.renamedHeaders && parsed.renamedHeaders.length > 0) {
                alert(`Duplicate column name(s) were renamed so every column is unique:\n` +
                    parsed.renamedHeaders.map(r => `• column ${r.column}: "${r.from}" → "${r.to}"`).join('\n'));
            }
            if (parsed.raggedRowCount > 0) {
                alert(`${parsed.raggedRowCount} row(s) have a different number of cells from the header row. Short rows were padded with blanks (treated as missing) and extra cells were ignored. Please check the file's delimiters and quoting.`);
            }

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
            if (DOM.pipelineDirectLaunchBtn) {
                DOM.pipelineDirectLaunchBtn.disabled = false;
            }

            // Sync with pipeline tab if currently active or loaded
            if (DOM.pipelineDatasetName) {
                DOM.pipelineDatasetName.innerText = name;
            }
            if (AppState.currentStep === 'pipeline') {
                initPipelineUI();
                goToPipelineSubstep(1);
            }
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
                <span class="stat-value" style="font-size: 16px; overflow:hidden; text-overflow:ellipsis;">${escapeHtml(AppState.fileName)}</span>
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
                        ${headers.map(h => `<th>${escapeHtml(h)}</th>`).join('')}
                    </tr>
                </thead>
                <tbody>
                    ${previewRows.map(r => `
                        <tr>
                            ${headers.map(h => {
                                const val = r[h];
                                const isMiss = DataEngine.isMissing(val);
                                return `<td style="${isMiss ? 'color: var(--accent-rose); font-style: italic;' : ''}">${isMiss ? '∅ (Empty)' : escapeHtml(val)}</td>`;
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
        headers.forEach((col, colIdx) => {
            const a = analysis[col];
            const cfg = AppState.columnConfigs[col];

            gridHtml += `
                <div class="var-card ${cfg.role === 'target' ? 'is-target' : (cfg.role === 'meta' ? 'is-meta' : '')}" id="card-col-${escapeHtml(col)}">
                    <div class="var-header">
                        <span class="var-name" title="${escapeHtml(col)}">${escapeHtml(col)}</span>
                        ${a.missingCount > 0 ? `<span class="pill pill-missing">${a.missingCount} missing (${a.missingPercent}%)</span>` : `<span class="pill" style="background: rgba(16, 185, 129, 0.1); color: #10b981;">Complete</span>`}
                    </div>

                    <div style="font-size: 11.5px; color: var(--text-muted);">
                        ${a.isNumeric ? `Numeric (range: ${a.stats.min} to ${a.stats.max})` : `Distinct values: ${a.distinctCount} (e.g. ${escapeHtml(a.distinctValues.slice(0, 3).join(', '))})`}
                    </div>

                    <div class="var-row-control">
                        <label for="var-role-${colIdx}">Variable Role</label>
                        <select id="var-role-${colIdx}" class="control-select var-role-select" data-col="${escapeHtml(col)}">
                            <option value="feature" ${cfg.role === 'feature' ? 'selected' : ''}>Predictor Feature</option>
                            <option value="target" ${cfg.role === 'target' ? 'selected' : ''}>Target (Outcome to Predict)</option>
                            <option value="meta" ${cfg.role === 'meta' ? 'selected' : ''}>Meta / ID (Ignore from Model)</option>
                        </select>
                    </div>

                    <div class="var-row-control">
                        <label for="var-type-${colIdx}">Variable Type</label>
                        <select id="var-type-${colIdx}" class="control-select var-type-select" data-col="${escapeHtml(col)}">
                            <option value="continuous" ${cfg.type === 'continuous' ? 'selected' : ''}>Continuous (Numerical / Scale)</option>
                            <option value="ordinal" ${cfg.type === 'ordinal' ? 'selected' : ''}>Ordinal (Ranked Categorical)</option>
                            <option value="nominal" ${cfg.type === 'nominal' ? 'selected' : ''}>Nominal (Unordered Categorical)</option>
                        </select>
                    </div>
                    ${renderTargetAndOrdinalControls(col, colIdx, a, cfg)}
                    ${cfg.role === 'feature' && cfg.type === 'nominal' && (a.distinctCount > 20 || (a.validCount > 0 && a.distinctCount / a.validCount > 0.5))
                        ? `<div style="font-size: 11px; color: var(--accent-amber);">⚠️ High cardinality (${a.distinctCount} levels): one-hot encoding adds a column per level, which slows training and invites overfitting. If this is an identifier, set it to Meta / ID; otherwise consider grouping rare levels.</div>`
                        : ''}
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
                // Re-render so target-only controls (positive class) appear/disappear
                renderVariableConfigGrid();
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
                // Re-render so ordinal / positive-class controls reflect the new type
                renderVariableConfigGrid();
            });
        });

        document.querySelectorAll('.var-positive-select').forEach(sel => {
            sel.addEventListener('change', (e) => {
                AppState.columnConfigs[e.target.dataset.col].positiveClass = e.target.value;
            });
        });

        document.querySelectorAll('.var-ordinal-order').forEach(input => {
            input.addEventListener('change', (e) => {
                const col = e.target.dataset.col;
                const levels = e.target.value.split(',').map(v => v.trim()).filter(v => v.length > 0);
                const known = (AppState.columnAnalysis[col]?.distinctValues || []).map(v => String(v).trim().toLowerCase());
                const unknown = levels.filter(l => !known.includes(l.toLowerCase()));
                AppState.columnConfigs[col].ordinalOrder = levels.length > 0 ? levels : undefined;
                const hint = document.getElementById(e.target.id + '-hint');
                if (hint) {
                    hint.textContent = unknown.length > 0
                        ? `⚠️ Not found in the data (ignored): ${unknown.join(', ')}`
                        : (levels.length > 0 ? '✓ Custom order saved' : 'Using automatic order');
                }
            });
        });
    }

    // Extra Step 2 controls: positive (event) class for a binary target, level order for ordinals
    function renderTargetAndOrdinalControls(col, colIdx, a, cfg) {
        let html = '';
        const levels = (a.distinctValues || []).map(v => String(v));
        if (cfg.role === 'target' && cfg.type !== 'continuous' && levels.length === 2) {
            const sorted = [...levels].sort();
            const positive = cfg.positiveClass !== undefined && levels.includes(String(cfg.positiveClass)) ? String(cfg.positiveClass) : sorted[1];
            html += `
                <div class="var-row-control">
                    <label for="var-positive-${colIdx}">Positive (event) class</label>
                    <select id="var-positive-${colIdx}" class="control-select var-positive-select" data-col="${escapeHtml(col)}">
                        ${sorted.map(v => `<option value="${escapeHtml(v)}" ${v === positive ? 'selected' : ''}>${escapeHtml(v)}</option>`).join('')}
                    </select>
                </div>
                <div style="font-size: 11px; color: var(--text-muted);">Used as "positive" for ROC-AUC, Brier score, sensitivity, PPV and TP/FN.</div>
            `;
        }
        if (cfg.type === 'ordinal' && cfg.role !== 'meta' && levels.length > 1) {
            const autoOrder = DataEngine.sortOrdinalLevels(levels);
            const current = Array.isArray(cfg.ordinalOrder) && cfg.ordinalOrder.length ? cfg.ordinalOrder : autoOrder;
            html += `
                <div class="var-row-control" style="flex-direction: column; align-items: stretch; gap: 4px;">
                    <label for="var-order-${colIdx}">Level order (lowest &rarr; highest, comma-separated)</label>
                    <input type="text" id="var-order-${colIdx}" class="control-input var-ordinal-order" data-col="${escapeHtml(col)}" value="${escapeHtml(current.join(', '))}">
                    <span id="var-order-${colIdx}-hint" style="font-size: 11px; color: var(--text-muted);">${Array.isArray(cfg.ordinalOrder) && cfg.ordinalOrder.length ? '✓ Custom order saved' : 'Automatic order: please check it matches the clinical order'}</span>
                </div>
            `;
        }
        return html;
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

            if (cfg.role === 'target') {
                // Outcomes are never imputed: rows with a missing outcome are always excluded
                rowsHtml += `
                    <tr>
                        <td style="font-weight: 600;">${escapeHtml(col)}</td>
                        <td><span class="pill pill-target">target</span></td>
                        <td><span class="pill pill-${cfg.type}">${cfg.type}</span></td>
                        <td><span class="pill pill-missing">${a.missingCount} / ${a.total} (${a.missingPercent}%)</span></td>
                        <td colspan="2" style="font-size: 12px; color: var(--text-secondary);">
                            Rows with a missing outcome are <strong>excluded</strong> (an outcome is never imputed).
                        </td>
                    </tr>
                `;
                return;
            }

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
                    <option value="mode" ${cfg.imputationStrategy === 'mode' ? 'selected' : ''}>Mode / Most Frequent (${escapeHtml(a.stats.mode)})</option>
                    <option value="median" ${cfg.imputationStrategy === 'median' ? 'selected' : ''}>${a.isNumeric ? `Median Rank (${a.stats.median})` : 'Median Rank (text levels: falls back to mode)'}</option>
                    <option value="constant" ${cfg.imputationStrategy === 'constant' ? 'selected' : ''}>Constant Rank...</option>
                    <option value="drop" ${cfg.imputationStrategy === 'drop' ? 'selected' : ''}>Drop Rows with Missing</option>
                `;
            } else {
                strategyOptionsHtml = `
                    <option value="mode" ${cfg.imputationStrategy === 'mode' ? 'selected' : ''}>Mode / Most Frequent ("${escapeHtml(a.stats.mode)}")</option>
                    <option value="constant" ${cfg.imputationStrategy === 'constant' ? 'selected' : ''}>New Category ("Missing" / "Unknown")</option>
                    <option value="drop" ${cfg.imputationStrategy === 'drop' ? 'selected' : ''}>Drop Rows with Missing</option>
                `;
            }

            rowsHtml += `
                <tr>
                    <td style="font-weight: 600;">${escapeHtml(col)}</td>
                    <td><span class="pill pill-${cfg.role}">${cfg.role}</span></td>
                    <td><span class="pill pill-${cfg.type}">${cfg.type}</span></td>
                    <td><span class="pill pill-missing">${a.missingCount} / ${a.total} (${a.missingPercent}%)</span></td>
                    <td>
                        <select class="control-select impute-strat-select" data-col="${escapeHtml(col)}" style="min-width: 200px;">
                            ${strategyOptionsHtml}
                        </select>
                    </td>
                    <td>
                        <input type="text" class="control-input impute-custom-val" data-col="${escapeHtml(col)}" placeholder="${cfg.type === 'continuous' ? '0' : 'Missing'}" value="${escapeHtml(cfg.customImputeVal || '')}" style="${cfg.imputationStrategy === 'constant' ? 'display:inline-block; width: 110px;' : 'display:none; width: 110px;'}">
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
                const customInput = document.querySelector(`.impute-custom-val[data-col="${cssEscape(col)}"]`);
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
                            <li>${a.type === 'drop_rows'
                                ? `<strong>Rows</strong>: ${escapeHtml(a.description)}`
                                : `<strong>${escapeHtml(a.column)}</strong>: ${escapeHtml(a.strategy)} &rarr; ${a.imputedCount} value(s) imputed (fill: ${escapeHtml(a.filledValue)})`}</li>
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
                // "Custom" still showing 42 would be indistinguishable from Default: propose a random seed
                const currentVal = (cardInput && cardInput.value) || (hpInput && hpInput.value) || '42';
                if (String(currentVal).trim() === '42') {
                    const newSeed = Math.floor(Math.random() * 90000) + 10000;
                    if (cardInput) cardInput.value = newSeed;
                    if (hpInput) hpInput.value = newSeed;
                    showSeedToast(`Custom seed for ${MODEL_INFO[mId]?.name || mId}: ${newSeed} (edit if required)`);
                }
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
                    if (splitInput && String(splitInput.value).trim() === '42') {
                        splitInput.value = Math.floor(Math.random() * 90000) + 10000;
                        showSeedToast(`Custom split seed: ${splitInput.value} (edit if required)`);
                    }
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
            const dropoutVal = parseFloat(document.getElementById('nn-dropout-input')?.value);
            hp.dropoutRate = isNaN(dropoutVal) ? 0.1 : Math.min(0.9, Math.max(0, dropoutVal)); // 0 is a valid choice
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
            const parsed = parseSeed(inputVal);
            seedVal = parsed !== null ? parsed : 42;
            // Keep the on-screen seed identical to the seed actually used
            if (parsed === null) {
                if (cardInput) cardInput.value = 42;
                if (hpInput) hpInput.value = 42;
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

        const watchdog = AppState.watchdogs ? AppState.watchdogs.step4 : null;

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

            if (watchdog) {
                watchdog.start(`Initializing tournament for ${modelsToTest.length} model(s)...`, modelsToTest.length, (reason) => {
                    console.warn("Step 4 Model Training stopped:", reason);
                    DOM.trainModelBtn.disabled = false;
                    DOM.trainStatus.innerHTML = `<span style="color: var(--accent-rose);">⚠️ Model training stopped: ${escapeHtml(reason || 'cancelled')}</span>`;
                });
            }

            // Clear any loss curve from a previous run so the chart never shows another model's history
            if (AppState.charts.loss) {
                AppState.charts.loss.destroy();
                AppState.charts.loss = null;
            }

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
                    const parsed = splitSeedInput ? parseSeed(splitSeedInput.value) : null;
                    splitSeed = parsed !== null ? parsed : 42;
                    if (parsed === null && splitSeedInput) splitSeedInput.value = 42; // show the seed actually used
                }

                // LEAK-FREE OPTION A: split the raw patient records FIRST, then fit imputation, encoding and
                // scaling on the training partition only and apply them unchanged to the test partition.
                const targetCol = prep.targetCol;
                const rawRows = AppState.parsedData.rows.filter(r => !DataEngine.isMissing(r[targetCol]) &&
                    (prep.isClassification || !isNaN(Number(r[targetCol]))));
                const rowSplit = DataEngine.trainTestSplit(
                    rawRows,
                    rawRows.map(r => String(r[targetCol]).trim()),
                    trainRatio,
                    prep.isClassification,
                    splitSeed
                );
                const trainImp = DataEngine.imputeDataset(rowSplit.trainX, AppState.columnConfigs, AppState.columnAnalysis);
                const testImp = DataEngine.imputeDataset(rowSplit.testX, AppState.columnConfigs, AppState.columnAnalysis, trainImp.plan);
                const splitPrep = DataEngine.prepareFeaturesAndTarget(trainImp.imputedRows, AppState.columnConfigs);
                const testData = DataEngine.prepareTestingDataset(testImp.imputedRows, splitPrep);
                const split = { trainX: splitPrep.X, trainY: splitPrep.y, testX: testData.X, testY: testData.y };
                AppState.trainTestData = split;

                if (split.testX.length === 0) {
                    throw new Error(`The test partition is empty (${rawRows.length} usable rows at a ${Math.round(trainRatio * 100)}/${testPct} split). Use a larger cohort, a smaller training percentage, or Option B.`);
                }
                if (split.trainX.length < 5) {
                    throw new Error(`Only ${split.trainX.length} training case(s) after splitting; at least 5 are required.`);
                }
                const splitThreshold = prep.isClassification ? null : trainingMedian(split.trainY);

                const shouldRetrain = DOM.retrainYes ? DOM.retrainYes.checked : true;

                for (let i = 0; i < modelsToTest.length; i++) {
                    if (watchdog && watchdog.aborted) throw new Error("Training aborted by user.");

                    const mId = modelsToTest[i];
                    const mInfo = MODEL_INFO[mId] || { name: mId.toUpperCase(), icon: '⚙️' };
                    const hyperparams = getHyperparamsForModel(mId, prep);

                    if (watchdog) {
                        watchdog.tick(i + 1, `Fitting ${mInfo.name} on ${Math.round(trainRatio * 100)}% split (${split.trainX.length} cases)...`);
                    }

                    DOM.trainStatus.innerHTML = `<span style="color: var(--accent-cyan);">🚀 [${i + 1}/${modelsToTest.length}] Training ${mInfo.name} on ${Math.round(trainRatio * 100)}% split (${split.trainX.length} cases &bull; Seed: ${hyperparams.seed})...</span>`;
                    await new Promise(r => setTimeout(r, 40));

                    if (modelsToTest.length === 1 && mId === 'nn') {
                        setupTrainingLossChart();
                    }

                    const valModel = ModelEngine.createModel(
                        mId,
                        prep.isClassification,
                        splitPrep.targetClasses.length || 2,
                        hyperparams
                    );

                    if (mId === 'nn') {
                        // Test split is never shown to the model: the NN carves its own validation split from the training data
                        await valModel.fit(split.trainX, split.trainY, null, null, (epoch, total, logs) => {
                            if (watchdog && watchdog.aborted) throw new Error("Training aborted by user."); // stops training at this epoch/tree
                            DOM.trainStatus.innerHTML = `<span style="color: var(--accent-cyan);">[${i + 1}/${modelsToTest.length}] ${mInfo.name}: Epoch ${epoch}/${total} &mdash; Train Loss: ${logs.loss.toFixed(4)} ${logs.val_loss !== undefined ? `| Internal Val Loss: ${logs.val_loss.toFixed(4)}` : ''}</span>`;
                            if (modelsToTest.length === 1) updateLossChart(epoch, logs.loss, logs.val_loss);
                            if (watchdog && (epoch % 4 === 0 || epoch === total)) {
                                watchdog.tick(i + 1, `${mInfo.name}: Epoch ${epoch}/${total} (Loss: ${logs.loss.toFixed(4)})`);
                            }
                        });
                    } else if (mId === 'ensemble') {
                        await valModel.fit(split.trainX, split.trainY, null, null, (curr, total, info) => {
                            if (watchdog && watchdog.aborted) throw new Error("Training aborted by user."); // stops training at this epoch/tree
                            DOM.trainStatus.innerHTML = `<span style="color: var(--accent-cyan);">[${i + 1}/${modelsToTest.length}] ${mInfo.name}: Sub-Model ${curr}/${total} (${info.stage || ''})...</span>`;
                            if (watchdog) {
                                watchdog.tick(i + 1, `${mInfo.name}: Sub-Model ${curr}/${total} (${info.stage || ''})`);
                            }
                        });
                    } else {
                        await fitWithYields(valModel, split.trainX, split.trainY, watchdog);
                    }

                    if (watchdog && watchdog.aborted) throw new Error("Training aborted by user.");

                    // Benchmark out-of-sample accuracy on held-out test split
                    const evalRes = ModelEngine.evaluate(valModel, split.testX, split.testY, prep.isClassification, splitPrep.targetClasses, {
                        regressionThreshold: splitThreshold,
                        thresholdSource: 'median of the training-split outcomes'
                    });
                    evalRes.protocol = 'option-a';
                    evalRes.testImputedCells = testImp.audit.filter(a => a.type === 'impute_column').reduce((acc, a) => acc + a.imputedCount, 0);
                    evalRes.testDroppedRows = rowSplit.testX.length - testImp.imputedRows.length;
                    evalRes.unknownClassCount = testData.unknownClassCount;
                    evalRes.retrainOn100 = shouldRetrain;
                    evalRes.trainRatio = trainRatio;
                    evalRes.trainPct = Math.round(trainRatio * 100);
                    evalRes.testPct = testPct;
                    evalRes.trainCount = split.trainX.length;
                    evalRes.testCount = split.testX.length;
                    // Rows actually available for the 100% retrain (after any 'drop rows' imputation choices)
                    evalRes.totalCount = prep.X.length;
                    evalRes.splitSeed = splitSeed;
                    evalRes.seed = hyperparams.seed;
                    if (valModel.trainingInfo) evalRes.nnNote = describeNNTraining(valModel.trainingInfo);
                    if (valModel.usedFallback) {
                        evalRes.fallbackNote = `TensorFlow.js training failed (${valModel.fallbackReason}); a linear (logistic/ridge) fallback model was evaluated instead of a neural network.`;
                    }
                    if (valModel.weightingNote) evalRes.weightingNote = valModel.weightingNote;
                    if (valModel.fallbackNote) evalRes.fallbackNote = valModel.fallbackNote;

                    let activeModelInstance = valModel;

                    if (shouldRetrain) {
                        if (watchdog) {
                            watchdog.tick(i + 1, `Retraining ${mInfo.name} on 100% of cohort (N=${prep.X.length})...`);
                        }
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
                                if (watchdog && watchdog.aborted) throw new Error("Training aborted by user."); // stops training at this epoch/tree
                                DOM.trainStatus.innerHTML = `<span style="color: var(--accent-cyan);">[${i + 1}/${modelsToTest.length}] ${mInfo.name} 100% Retrain: Epoch ${epoch}/${total} &mdash; Loss: ${logs.loss.toFixed(4)}</span>`;
                                if (watchdog && (epoch % 4 === 0 || epoch === total)) {
                                    watchdog.tick(i + 1, `${mInfo.name} 100% Retrain: Epoch ${epoch}/${total}`);
                                }
                            });
                        } else if (mId === 'ensemble') {
                            await fullModel.fit(prep.X, prep.y, null, null, (curr, total, info) => {
                                if (watchdog && watchdog.aborted) throw new Error("Training aborted by user."); // stops training at this epoch/tree
                                DOM.trainStatus.innerHTML = `<span style="color: var(--accent-cyan);">[${i + 1}/${modelsToTest.length}] ${mInfo.name} 100% Retrain: Base ${curr}/${total} (${info.stage || ''})...</span>`;
                                if (watchdog) {
                                    watchdog.tick(i + 1, `${mInfo.name} 100% Retrain: Base ${curr}/${total} (${info.stage || ''})`);
                                }
                            });
                        } else {
                            await fitWithYields(fullModel, prep.X, prep.y, watchdog);
                        }
                        activeModelInstance = fullModel;
                        // The split-phase model has been evaluated and is replaced: free its GPU memory
                        disposeModels([valModel]);
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
                // Predictions must use the encoding the deployed models were fitted with:
                // full-cohort encoding after 100% retraining, otherwise the training-split encoding.
                AppState.prepMeta = shouldRetrain ? prep : splitPrep;
            } else {
                // =============================================================
                // OPTION B: SEPARATE TRAINING (i) AND TESTING (ii) DATASETS
                // =============================================================
                if (!AppState.separateTestRawRows || AppState.separateTestRawRows.length === 0) {
                    DOM.trainStatus.innerHTML = `<span style="color: var(--accent-amber);">⚠️ Option B requires a separate testing cohort (Dataset ii). Please upload a test CSV or click 'Load Built-In Validation Cohort'.</span>`;
                    DOM.trainModelBtn.disabled = false;
                    if (watchdog) watchdog.forceAbort("Missing external test cohort.");
                    alert("Option B requires a separate testing cohort (Dataset ii). Please upload a test CSV or load the built-in validation cohort in the Option B card.");
                    return;
                }

                // Transform separate test data using training metadata (strict zero leakage)
                const testPrep = DataEngine.prepareTestingDataset(AppState.separateTestRawRows, prep);
                AppState.separateTestData = testPrep;

                if (!testPrep.totalEvaluated || testPrep.totalEvaluated === 0) {
                    DOM.trainStatus.innerHTML = `<span style="color: var(--accent-rose);">Testing cohort evaluation failed: 0 valid test cases found.</span>`;
                    DOM.trainModelBtn.disabled = false;
                    if (watchdog) watchdog.forceAbort("0 valid test cases in external cohort.");
                    alert(`Could not extract any valid test cases with matching predictors and target. Missing target cases: ${testPrep.missingTargetCount}, unknown classes: ${testPrep.unknownClassCount}. Please check the testing CSV.`);
                    return;
                }

                for (let i = 0; i < modelsToTest.length; i++) {
                    if (watchdog && watchdog.aborted) throw new Error("Training aborted by user.");

                    const mId = modelsToTest[i];
                    const mInfo = MODEL_INFO[mId] || { name: mId.toUpperCase(), icon: '⚙️' };
                    const hyperparams = getHyperparamsForModel(mId, prep);

                    if (watchdog) {
                        watchdog.tick(i + 1, `Fitting ${mInfo.name} on 100% of Dataset (i) (${prep.X.length} cases)...`);
                    }

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
                        // Dataset (ii) is never shown to the model: the NN carves its own validation split from Dataset (i)
                        await model.fit(prep.X, prep.y, null, null, (epoch, total, logs) => {
                            if (watchdog && watchdog.aborted) throw new Error("Training aborted by user."); // stops training at this epoch/tree
                            DOM.trainStatus.innerHTML = `<span style="color: var(--accent-purple);">[${i + 1}/${modelsToTest.length}] ${mInfo.name}: Epoch ${epoch}/${total} &mdash; Train Loss: ${logs.loss.toFixed(4)} ${logs.val_loss !== undefined ? `| Internal Val Loss: ${logs.val_loss.toFixed(4)}` : ''}</span>`;
                            if (modelsToTest.length === 1) updateLossChart(epoch, logs.loss, logs.val_loss);
                            if (watchdog && (epoch % 4 === 0 || epoch === total)) {
                                watchdog.tick(i + 1, `${mInfo.name}: Epoch ${epoch}/${total}`);
                            }
                        });
                    } else if (mId === 'ensemble') {
                        await model.fit(prep.X, prep.y, null, null, (curr, total, info) => {
                            if (watchdog && watchdog.aborted) throw new Error("Training aborted by user."); // stops training at this epoch/tree
                            DOM.trainStatus.innerHTML = `<span style="color: var(--accent-purple);">[${i + 1}/${modelsToTest.length}] ${mInfo.name}: Base ${curr}/${total} (${info.stage || ''})...</span>`;
                            if (watchdog) {
                                watchdog.tick(i + 1, `${mInfo.name}: Base ${curr}/${total} (${info.stage || ''})`);
                            }
                        });
                    } else {
                        await fitWithYields(model, prep.X, prep.y, watchdog);
                    }

                    if (watchdog && watchdog.aborted) throw new Error("Training aborted by user.");

                    // Evaluate on independent Dataset (ii)
                    const evalRes = ModelEngine.evaluate(model, testPrep.X, testPrep.y, prep.isClassification, prep.targetClasses, {
                        regressionThreshold: prep.isClassification ? null : trainingMedian(prep.y),
                        thresholdSource: 'median of the Dataset (i) training outcomes'
                    });
                    evalRes.protocol = 'option-b';
                    evalRes.retrainOn100 = false;
                    evalRes.trainCount = prep.X.length;
                    evalRes.testCount = testPrep.totalEvaluated;
                    evalRes.totalCount = prep.X.length + testPrep.totalEvaluated;
                    evalRes.testProvided = testPrep.totalProvided;
                    evalRes.missingTargetCount = testPrep.missingTargetCount;
                    evalRes.unknownClassCount = testPrep.unknownClassCount;
                    evalRes.imputedFeatureCells = testPrep.imputedFeatureCells;
                    evalRes.seed = hyperparams.seed;
                    if (model.trainingInfo) evalRes.nnNote = describeNNTraining(model.trainingInfo);
                    if (model.usedFallback) {
                        evalRes.fallbackNote = `TensorFlow.js training failed (${model.fallbackReason}); a linear (logistic/ridge) fallback model was evaluated instead of a neural network.`;
                    }
                    if (model.weightingNote) evalRes.weightingNote = model.weightingNote;
                    if (model.fallbackNote) evalRes.fallbackNote = model.fallbackNote;

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

            // Sort tested models by out-of-sample performance (macro-F1 then AUC for classification; R² for regression)
            testedResults.sort((a, b) => compareEvalResults(a.evalRes, b.evalRes, prep.isClassification));

            const previousResults = AppState.allTestedResults;
            AppState.allTestedResults = testedResults;
            // Designate top-performing model as champion
            const champion = testedResults[0];
            AppState.trainedModel = champion.model;
            AppState.deployedModelId = champion.modelId;
            AppState.evalResults = champion.evalRes;
            AppState.selectedModelId = champion.modelId;
            // Remember the encoding the Step 4 models use, then mark the standard workflow as the source
            AppState.baselinePrepMeta = AppState.prepMeta;
            markBaselineDeployed();
            // The previous run's models are no longer reachable (the new leader is now deployed)
            disposeModels((previousResults || []).map(r => r.model));

            if (watchdog) {
                watchdog.finish(`Tournament Complete: Evaluated ${testedResults.length} Models. Top-ranked on ${rankingMetricLabel(prep.isClassification)}: ${champion.info.name}`);
            }

            // Update train status banner
            if (testedResults.length === 1) {
                const singleRes = champion.evalRes;
                const accText = singleRes.isClassification
                    ? `Test Accuracy: ${(singleRes.accuracy * 100).toFixed(1)}% (ROC-AUC: ${singleRes.rocAuc !== null && singleRes.rocAuc !== undefined ? singleRes.rocAuc.toFixed(3) : 'N/A'})`
                    : `Test R²: ${fmtR2(singleRes.r2)} (RMSE: ${singleRes.rmse.toFixed(3)})`;
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
                    ? `Macro F1: ${(champion.evalRes.macroF1 * 100).toFixed(1)}%, Accuracy: ${(champion.evalRes.accuracy * 100).toFixed(1)}%`
                    : `R²: ${fmtR2(champion.evalRes.r2)}`;
                DOM.trainStatus.innerHTML = `
                    <div style="display: flex; flex-direction: column; gap: 3px;">
                        <span style="color: var(--accent-emerald); font-weight: 700;">
                            🏆 Tournament Complete: Tested &amp; Evaluated ${testedResults.length} Models!
                        </span>
                        <span style="font-size: 11.5px; color: var(--text-secondary);">
                            Top-ranked on ${escapeHtml(rankingMetricLabel(prep.isClassification))}: <strong>${champion.info.name}</strong> [${topAccText} &bull; Seed: <code style="color: var(--accent-purple); font-weight: 700;">🌱 ${champion.evalRes.seed !== null ? champion.evalRes.seed : 'Auto'}</code>] &bull; All model seeds are listed below.
                        </span>
                        <span style="font-size: 11px; color: var(--accent-amber);">
                            ⚠️ The leader was chosen using the same test cases it is scored on, so its test score is optimistically biased. Confirm the chosen model on an independent cohort (Option B).
                        </span>
                    </div>
                `;
            }

            DOM.trainModelBtn.disabled = false;
            DOM.toStep5Btn.disabled = false;
            DOM.toStep6Btn.disabled = false;

            renderEvaluationDashboard();
        } catch (e) {
            if (watchdog && !watchdog.aborted) {
                watchdog.forceAbort(`Training failed: ${e.message}`);
            }
            DOM.trainStatus.innerHTML = `<span style="color: var(--accent-rose);">Training halted: ${e.message}</span>`;
            DOM.trainModelBtn.disabled = false;
            if (e.message !== "Training aborted by user." && e.message !== "Operation halted by user.") {
                alert(`Training error: ${e.message}`);
            }
        }
    }

    // Shows the ranking metric that applies to the current outcome type in both selectors
    function syncRankingSelectors(isClassification) {
        const value = `${isClassification ? 'clf' : 'reg'}:${rankingMetricKey(isClassification)}`;
        const otherPrefix = isClassification ? 'reg:' : 'clf:';
        document.querySelectorAll('.ranking-metric-select').forEach(sel => {
            // Metrics for the other outcome type cannot be chosen (selecting one would have no effect
            // and the selector would snap back), so they are disabled while these results are shown
            Array.from(sel.options).forEach(opt => { opt.disabled = opt.value.startsWith(otherPrefix); });
            sel.value = value;
        });
    }

    // Frees GPU (WebGL) memory held by trained neural networks / ensembles that are no longer reachable.
    // The currently deployed model is never disposed.
    function disposeModels(models) {
        (models || []).forEach(m => {
            if (m && m !== AppState.trainedModel && typeof m.dispose === 'function') {
                try { m.dispose(); } catch (e) { /* ignore */ }
            }
        });
    }

    // Fits a non-neural model. Random Forest and logistic/ridge regression yield to the browser between
    // trees/iterations so the UI repaints and Force Stop can interrupt them; other models fit synchronously.
    async function fitWithYields(model, X, y, watchdog) {
        if (typeof model.fitAsync === 'function') {
            await model.fitAsync(X, y, {
                onYield: () => {
                    if (watchdog && watchdog.aborted) throw new Error("Training aborted by user.");
                }
            });
        } else {
            model.fit(X, y);
        }
    }

    // Re-sorts already-computed results after the ranking metric changes (no retraining, no redeployment)
    function rerankExistingResults() {
        if (AppState.allTestedResults && AppState.allTestedResults.length > 0 && AppState.prepMeta) {
            const isClf = AppState.allTestedResults[0].evalRes.isClassification;
            AppState.allTestedResults.sort((a, b) => compareEvalResults(a.evalRes, b.evalRes, isClf));
            if (AppState.evalResults) renderEvaluationDashboard();
        }
        if (AppState.pipeline && AppState.pipeline.benchmarkResults && AppState.pipeline.benchmarkResults.length > 0) {
            const isClf = AppState.pipeline.columnConfigs[AppState.pipeline.targetCol]?.type !== 'continuous';
            AppState.pipeline.benchmarkResults.sort((a, b) => compareEvalResults(a.eval, b.eval, isClf));
            renderPipelineLeaderboardOnly();
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
                        label: 'Internal Validation Loss (held out from training data)',
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
                    legend: { labels: { color: isDark ? '#f8fafc' : '#0f172a' } },
                    title: {
                        display: true,
                        // The optional 100% retrain is not plotted
                        text: 'Evaluation-phase fit (single neural network run); 100% retrain not plotted',
                        color: isDark ? '#94a3b8' : '#64748b',
                        font: { size: 11, weight: 'normal' }
                    }
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
        markBaselineDeployed();

        // Re-render dashboard to refresh badges and active buttons
        renderEvaluationDashboard();

        // Update single prediction form headers if open
        if (AppState.prepMeta) {
            renderSinglePredictionForm();
        }

        showSeedToast(`✓ Deployed "${displayModelName(modelId, targetItem.model)}" as the active predictor for Steps 5 & 6`);
    }

    // A Step 4 (standard workflow) model is now active: record that, and restore the encoding it was
    // trained with (a previously deployed pipeline model may have replaced AppState.prepMeta)
    function markBaselineDeployed() {
        AppState.deployedSource = 'standard';
        AppState.deployedPipelineModel = null;
        if (AppState.baselinePrepMeta) AppState.prepMeta = AppState.baselinePrepMeta;
        if (typeof updatePipelineActiveDeploymentStatus === 'function') updatePipelineActiveDeploymentStatus();
        if (typeof updatePredictionProvenanceBar === 'function') updatePredictionProvenanceBar();
        if (AppState.pipeline && AppState.pipeline.benchmarkResults && AppState.pipeline.benchmarkResults.length > 0 && DOM.pipelineLeaderboardContainer) {
            renderPipelineLeaderboardOnly();
        }
    }
    window.deployTestedModel = deployTestedModel;

    function showSeedToast(msg) {
        const existing = document.querySelector('.seed-toast');
        if (existing) existing.remove();
        const toast = document.createElement('div');
        toast.className = 'seed-toast';
        toast.setAttribute('role', 'status');
        toast.innerHTML = '<span>📋</span> <span></span>';
        toast.lastElementChild.textContent = msg;
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
        syncRankingSelectors(evalRes.isClassification);
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

        // Methodological notes that affect how the metrics should be read
        if (DOM.retrainStatusBanner) {
            const notes = [];
            const resultsForNotes = allResults.length ? allResults.map(r => r.evalRes) : [evalRes];
            if (prep && prep.isClassification && prep.positiveClass !== null && prep.positiveClass !== undefined) {
                notes.push(`Positive (event) class for ROC-AUC, Brier score, sensitivity and TP/FN: <strong>${escapeHtml(prep.positiveClass)}</strong>. Change it in Step 2 (target card) if this is not the outcome of interest.`);
            }
            if (isOptionB && evalRes.imputedFeatureCells > 0) {
                notes.push(`${evalRes.imputedFeatureCells} missing/unrecognised predictor cell(s) in Dataset (ii) were filled from training-cohort statistics.`);
            }
            if (isOptionB && (evalRes.missingTargetCount > 0 || evalRes.unknownClassCount > 0)) {
                notes.push(`Dataset (ii) rows excluded: ${evalRes.missingTargetCount || 0} with a missing outcome, ${evalRes.unknownClassCount || 0} with an outcome class not seen in training.`);
            }
            if (!isOptionB) {
                notes.push('Option A: records were split first; imputation, category encoding and scaling were fitted on the training split only and applied unchanged to the test split. The split is random at row level: if patients contribute several rows, or the data are temporal or multicentre, use a patient-level, temporal or external split (Option B) instead.');
                if (evalRes.testImputedCells > 0) {
                    notes.push(`${evalRes.testImputedCells} missing predictor value(s) in the test split were filled with training-derived values.`);
                }
                if (evalRes.unknownClassCount > 0) {
                    notes.push(`${evalRes.unknownClassCount} test row(s) had an outcome class absent from the training split and were excluded.`);
                }
            }
            if (prep && prep.isClassification) {
                notes.push(`Class "probabilities" are uncalibrated: Random Forest and hard voting give vote shares, k-NN gives neighbour shares. The Brier score (lower is better) summarises probability accuracy; a calibration plot is recommended before clinical use.`);
            }
            summariseHeldOutSet(evalRes).forEach(n => notes.push(n));
            resultsForNotes.forEach(r => {
                if (r && r.fallbackNote) notes.push(`⚠️ ${escapeHtml(r.fallbackNote)}`);
                if (r && r.weightingNote) notes.push(`Ensemble: ${escapeHtml(r.weightingNote)}`);
                if (r && r.nnNote) notes.push(`Neural network: ${escapeHtml(r.nnNote)}`);
            });
            if (notes.length > 0) {
                DOM.retrainStatusBanner.innerHTML += `
                    <div style="font-size: 11.5px; color: var(--text-secondary); margin: -6px 0 14px 0; padding: 8px 12px; border-left: 3px solid var(--accent-amber); background: var(--bg-surface-subtle); border-radius: var(--radius-sm);">
                        ${Array.from(new Set(notes)).map(n => `<div>• ${n}</div>`).join('')}
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

                    const rankBadge = isChampion ? '🥇 #1' : (idx === 1 ? '🥈 #2' : (idx === 2 ? '🥉 #3' : `#${idx + 1}`));
                    const scoreDisplay = rEval.isClassification 
                        ? `<strong style="color: var(--accent-cyan); font-size: 14.5px;">${(rEval.accuracy * 100).toFixed(1)}%</strong>${Array.isArray(rEval.accuracyCI) && rEval.accuracyCI.length === 2 ? `<div style="font-size: 10px; color: var(--text-muted);">95% CI ${(rEval.accuracyCI[0] * 100).toFixed(0)}–${(rEval.accuracyCI[1] * 100).toFixed(0)}%</div>` : ''}`
                        : `<strong style="color: var(--accent-cyan); font-size: 14.5px;">R²: ${fmtR2(rEval.r2)}</strong>`;

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
                    const balDisplay = rEval.isClassification && typeof rEval.balancedAccuracy === 'number'
                        ? `${(rEval.balancedAccuracy * 100).toFixed(1)}%` : '-';
                    const brierDisplay = rEval.isClassification && typeof rEval.brierScore === 'number'
                        ? rEval.brierScore.toFixed(3) : '-';

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
                                        <div style="font-weight: 700; color: var(--text-primary);">${escapeHtml(displayModelName(resItem.modelId, resItem.model))}</div>
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
                            <td>${balDisplay}</td>
                            <td>${brierDisplay}</td>
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
                                    <span>🏆</span> Benchmark Leaderboard &mdash; ${allResults.length} Tested Models Ranked by ${escapeHtml(rankingMetricLabel(prep.isClassification))}
                                </h3>
                                <div style="font-size: 12px; color: var(--text-secondary); margin-top: 2px;">
                                    Comparative performance on ${cohortDesc}, ranked by <strong>${escapeHtml(rankingMetricLabel(prep.isClassification))}</strong> (change the primary metric above). The top-ranked model (${allResults[0].info.name}) is set as the active predictor. A single train/test split does not show that one model is generally superior, and because the leader was selected on these same test cases its score is optimistically biased. Report all metrics, not only the winner. Click any seed pill to copy it.
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
                                        <th>${prep.isClassification ? 'Balanced Acc' : '-'}</th>
                                        <th>${prep.isClassification ? 'Brier ↓' : '-'}</th>
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
                                            ${classes.map(c => `<th style="padding: 6px 8px;">${escapeHtml(c)}</th>`).join('')}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        ${classes.map((actualClass, r) => `
                                            <tr>
                                                <th style="text-align: left; padding: 6px 8px;">${escapeHtml(actualClass)}</th>
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
                        cmContent = `
                            ${renderExploratoryCutoffTable(rEval, true)}
                            <div style="display: flex; justify-content: space-between; margin-top: 6px; font-size: 11px; color: var(--text-secondary); padding: 0 4px;">
                                <span>RMSE: <strong>${rEval.rmse.toFixed(3)}</strong></span>
                                <span>MAE: <strong>${rEval.mae.toFixed(3)}</strong></span>
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
                                        <div class="cm-card-title">${escapeHtml(displayModelName(resItem.modelId, resItem.model))}</div>
                                        <span class="pill ${mInfo.pillClass}" style="font-size: 10px; padding: 1px 6px;">${mInfo.tag}</span>
                                    </div>
                                </div>
                                <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 4px;">
                                    ${isChampion ? `<span class="champion-badge">🏆 Top-ranked (${escapeHtml(rankingMetricLabel(prep.isClassification))})</span>` : `<span class="pill" style="font-size: 10.5px;">Rank #${idx + 1}</span>`}
                                    ${isDeployed ? `<span class="pill pill-target" style="font-size: 10px; background: rgba(16, 185, 129, 0.15); color: var(--accent-emerald);">⭐ Active</span>` : ''}
                                </div>
                            </div>

                            <div class="cm-card-metrics">
                                <div class="cm-metric-item">
                                    <span class="cm-m-label">${rEval.isClassification ? 'Test Accuracy' : 'R² Score'}</span>
                                    <span class="cm-m-val">${rEval.isClassification ? `${(rEval.accuracy * 100).toFixed(1)}%` : fmtR2(rEval.r2)}</span>
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
                                ${rEval.isClassification ? 'Confusion Matrix:' : 'Error metrics &amp; exploratory 2×2 split:'}
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
                        ${Array.isArray(evalRes.accuracyCI) && evalRes.accuracyCI.length === 2 ? `<span style="font-size: 11px; color: var(--text-muted);">95% CI ${(evalRes.accuracyCI[0] * 100).toFixed(1)}–${(evalRes.accuracyCI[1] * 100).toFixed(1)}% (Wilson)</span>` : ''}
                    </div>
                    ${typeof evalRes.balancedAccuracy === 'number' ? `
                    <div class="metric-pill-card">
                        <span class="m-title">Balanced Accuracy</span>
                        <span class="m-val">${(evalRes.balancedAccuracy * 100).toFixed(1)}%</span>
                    </div>` : ''}
                    ${typeof evalRes.brierScore === 'number' ? `
                    <div class="metric-pill-card">
                        <span class="m-title">Brier Score (lower is better)</span>
                        <span class="m-val">${evalRes.brierScore.toFixed(3)}</span>
                    </div>` : ''}
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
                                        ${classes.map(c => `<th>${escapeHtml(c)}</th>`).join('')}
                                    </tr>
                                </thead>
                                <tbody>
                                    ${classes.map((actualClass, r) => `
                                        <tr>
                                            <th style="text-align: left;">${escapeHtml(actualClass)}</th>
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
                        <span class="m-val">${fmtR2(evalRes.r2)}</span>
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
                    <div class="metric-pill-card" style="cursor: pointer;" onclick="window.copySeedToClipboard(${evalRes.seed})" title="Click to copy reproducibility seed: ${evalRes.seed}">
                        <span class="m-title">Reproducibility Seed</span>
                        <span class="m-val" style="color: var(--accent-purple); font-family: monospace;">🌱 ${evalRes.seed !== null && evalRes.seed !== undefined ? evalRes.seed : 'Auto'} 📋</span>
                    </div>
                `;

                const regCmHtml = renderExploratoryCutoffTable(evalRes, false);

                DOM.evaluationDetailContainer.innerHTML = `
                    ${regCmHtml}
                    <p style="font-size: 13px; color: var(--text-secondary); margin-top: 4px;">
                        Model validated on ${cohortDesc} with R² = ${fmtR2(evalRes.r2)} and Mean Absolute Error = ${evalRes.mae}. ${isOptionB ? `Fitted strictly on 100% of Training Dataset (i) (${trainCount} patients).` : (evalRes.retrainOn100 ? `Active model was then retrained on 100% of data (${totalCount} patients).` : `Active model maintained strictly on ${trainPct}% training split (${trainCount} patients; 100% retraining declined).`)}
                    </p>
                    <div style="font-size: 11.5px; color: var(--text-muted); margin-top: 6px;">
                        💡 <em>R², RMSE and MAE are the primary performance measures for a continuous outcome. If the outcome has a prespecified clinical decision threshold, analyse it as a binary outcome (Nominal) instead of relying on the exploratory split above.</em>
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

        featureCols.forEach((col, colIdx) => {
            const meta = featureMetadata[col];
            const fieldId = `predict-field-${colIdx}`;

            if (meta.type === 'continuous') {
                const step = (meta.max - meta.min) > 20 ? '1' : '0.01';
                formHtml += `
                    <div class="form-field">
                        <label for="${fieldId}">
                            <span>${escapeHtml(col)}</span>
                            <span class="field-hint">Range: ${meta.min} &ndash; ${meta.max}</span>
                        </label>
                        <input type="number" step="any" id="${fieldId}" class="control-input single-input-field" data-col="${escapeHtml(col)}" value="${meta.mean.toFixed(2)}">
                    </div>
                `;
            } else if (meta.type === 'ordinal') {
                formHtml += `
                    <div class="form-field">
                        <label for="${fieldId}">
                            <span>${escapeHtml(col)}</span>
                            <span class="field-hint">Ordinal Rank</span>
                        </label>
                        <select id="${fieldId}" class="control-select single-input-field" data-col="${escapeHtml(col)}">
                            ${meta.ranks.map(rk => `<option value="${escapeHtml(rk)}">${escapeHtml(rk)}</option>`).join('')}
                        </select>
                    </div>
                `;
            } else {
                // Nominal
                formHtml += `
                    <div class="form-field">
                        <label for="${fieldId}">
                            <span>${escapeHtml(col)}</span>
                            <span class="field-hint">Categorical</span>
                        </label>
                        <select id="${fieldId}" class="control-select single-input-field" data-col="${escapeHtml(col)}">
                            ${meta.categories.map(cat => `<option value="${escapeHtml(cat)}">${escapeHtml(cat)}</option>`).join('')}
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

        const inputIssues = [];
        const xVector = DataEngine.transformSingleRow(inputRow, AppState.prepMeta, inputIssues);
        const result = AppState.trainedModel.predictSingle(xVector);
        const prep = AppState.prepMeta;
        const inputWarningHtml = inputIssues.length > 0
            ? `<div style="font-size: 11.5px; color: var(--accent-amber); margin-top: 6px; font-weight: 600;">
                    ⚠️ Blank or unrecognised input(s) replaced with training-cohort values (mean / median rank / most frequent category): ${escapeHtml(inputIssues.map(i => i.column).join(', '))}
               </div>`
            : '';

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
            const predictedLabel = prep.targetIdxToClass[result.pred] !== undefined ? prep.targetIdxToClass[result.pred] : result.pred;
            const hasConfidence = typeof result.confidence === 'number' && !isNaN(result.confidence);
            const confidencePercent = hasConfidence ? (result.confidence * 100).toFixed(1) : null;

            let probBarsHtml = '';
            if (result.probabilities) {
                probBarsHtml = Object.keys(result.probabilities).map(clsIdx => {
                    const clsLabel = prep.targetIdxToClass[clsIdx] !== undefined ? prep.targetIdxToClass[clsIdx] : clsIdx;
                    const p = (result.probabilities[clsIdx] * 100).toFixed(1);
                    return `
                        <div style="font-size: 12px; display: flex; justify-content: space-between; gap: 10px;">
                            <span>${escapeHtml(clsLabel)}:</span>
                            <strong>${p}%</strong>
                        </div>
                    `;
                }).join('');
            }

            DOM.singleResultBanner.innerHTML = `
                <div class="banner-left">
                    <h3>Predicted ${escapeHtml(prep.targetCol)}</h3>
                    <div class="pred-outcome-val">${escapeHtml(predictedLabel)}</div>
                    <div style="font-size: 13px; color: var(--text-secondary); margin-top: 6px;">
                        Active Model: <strong>${escapeHtml(displayModelName(AppState.deployedModelId || AppState.selectedModelId, AppState.trainedModel))}</strong> &bull; Target: <em>${escapeHtml(prep.targetCol)}</em>${seedSnippet}
                    </div>
                    ${ensembleDetailsHtml}
                    ${inputWarningHtml}
                    <div style="font-size: 11px; color: var(--text-muted); margin-top: 6px;">
                        ${probabilityLabel(result.probabilityType).note}: not a calibrated clinical risk or a measure of certainty (check the Brier score in Step 4). For research and educational use only.
                    </div>
                </div>
                <div class="banner-confidence">
                    <span style="font-size: 12px; color: var(--text-muted); text-transform: uppercase;">${probabilityLabel(result.probabilityType).title}</span>
                    <span style="font-size: 26px; font-weight: 800; color: var(--accent-emerald);">${hasConfidence ? `${confidencePercent}%` : 'N/A'}</span>
                    <div class="prob-meter-bar">
                        <div class="prob-meter-fill" style="width: ${hasConfidence ? confidencePercent : 0}%;"></div>
                    </div>
                    <div style="margin-top: 8px;">
                        ${probBarsHtml}
                    </div>
                </div>
            `;
        } else {
            DOM.singleResultBanner.innerHTML = `
                <div class="banner-left">
                    <h3>Predicted Continuous ${escapeHtml(prep.targetCol)}</h3>
                    <div class="pred-outcome-val">${result.pred}</div>
                    <div style="font-size: 13px; color: var(--text-secondary); margin-top: 6px;">
                        Active Model: <strong>${escapeHtml(displayModelName(AppState.deployedModelId || AppState.selectedModelId, AppState.trainedModel))}</strong> &bull; Unit/Scale: <em>${escapeHtml(prep.targetCol)}</em>${seedSnippet}
                    </div>
                    ${ensembleDetailsHtml}
                    ${inputWarningHtml}
                </div>
            `;
        }
    }

    // =========================================================================
    // STEP 6: BATCH PREDICTION FACILITY
    // =========================================================================
    function handleBatchFileUpload(file) {
        if (!file.name.toLowerCase().endsWith('.csv') && file.type !== 'text/csv') {
            alert("Please upload a valid .csv file for batch prediction.");
            return;
        }
        if (file.size > 50 * 1024 * 1024) {
            alert("The batch file exceeds 50 MB. Please split it into smaller files (all processing happens in the browser).");
            return;
        }
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

        const batchModelName = displayModelName(AppState.deployedModelId || AppState.selectedModelId, AppState.trainedModel);
        // Predictor columns the model needs but the batch file lacks entirely
        const absentCols = prep.featureCols.filter(c => !headers.includes(c));
        let rowsWithImputation = 0;
        let totalImputedCells = 0;

        // Encode every row, then predict in one batched pass (one GPU call for neural networks)
        const rowIssueLists = rows.map(() => []);
        const xMatrix = rows.map((r, i) => DataEngine.transformSingleRow(r, prep, rowIssueLists[i]));
        const outputs = rows.length ? ModelEngine.predictOutputs(AppState.trainedModel, xMatrix) : [];

        rows.forEach((r, rowIdx) => {
            const rowIssues = rowIssueLists[rowIdx];
            const predRes = outputs[rowIdx];

            let predictedLabel = predRes.pred;
            let confidence = 'N/A';
            if (prep.isClassification) {
                predictedLabel = prep.targetIdxToClass[predRes.pred] !== undefined ? prep.targetIdxToClass[predRes.pred] : predRes.pred;
                confidence = (typeof predRes.confidence === 'number' && !isNaN(predRes.confidence))
                    ? `${(predRes.confidence * 100).toFixed(1)}%` : 'N/A';
            }
            if (rowIssues.length > 0) {
                rowsWithImputation++;
                totalImputedCells += rowIssues.length;
            }

            const rowCopy = { ...r };
            rowCopy[`Predicted_${prep.targetCol}`] = predictedLabel;
            rowCopy[`Prediction_Probability`] = confidence;
            rowCopy[`Probability_Type`] = prep.isClassification ? probabilityLabel(predRes.probabilityType).title : 'N/A (continuous outcome)';
            rowCopy[`Imputed_Inputs`] = rowIssues.length > 0 ? rowIssues.map(i => `${i.column}:${i.kind}`).join('; ') : '';
            rowCopy[`Model`] = batchModelName;
            rowCopy[`Model_Seed`] = activeSeed;
            results.push(rowCopy);
        });

        if (results.length === 0) {
            alert("The batch file contains no data rows.");
            return;
        }
        if (rowsWithImputation > 0 || absentCols.length > 0) {
            alert(`Batch prediction completed with substituted inputs:\n` +
                (absentCols.length > 0 ? `• Columns missing from the file (filled for every row): ${absentCols.join(', ')}\n` : '') +
                `• ${rowsWithImputation} of ${results.length} row(s) had ${totalImputedCells} blank or unrecognised predictor value(s) replaced with training-cohort values.\n` +
                `See the 'Imputed_Inputs' column. Predictions for these rows are less reliable.`);
        }

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
                        ${resultHeaders.map(h => `<th style="${h.startsWith('Predicted_') ? 'color: var(--accent-cyan); background: rgba(6,182,212,0.1);' : ''}">${escapeHtml(h)}</th>`).join('')}
                    </tr>
                </thead>
                <tbody>
                    ${previewSlice.map(r => `
                        <tr>
                            ${resultHeaders.map(h => `
                                <td style="${h.startsWith('Predicted_') ? 'font-weight: 700; color: var(--accent-cyan); background: rgba(6,182,212,0.05);' : ''}">${escapeHtml(r[h])}</td>
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
        const lines = [headers.map(h => (/[",\r\n]/.test(h) ? `"${h.replace(/"/g, '""')}"` : h)).join(',')];

        AppState.batchPredictions.forEach(row => {
            const line = headers.map(h => {
                let val = String(row[h] !== undefined ? row[h] : '');
                // Prevent spreadsheet formula execution from cell contents
                if (/^[=+\-@\t\r]/.test(val) && isNaN(Number(val))) val = `'${val}`;
                if (/[",\r\n]/.test(val)) {
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

    // =========================================================================
    // 7-STEP CLINICAL ML PIPELINE CONTROLLER
    // Conceived, designed and developed by Dr G Narenthiran MB ChB BSc(MedSci)(Hons) FEBNS FRCS(SN)
    // =========================================================================

    function initPipelineUI() {
        if (DOM.pipelineDatasetName) {
            DOM.pipelineDatasetName.innerText = AppState.fileName || "No dataset loaded yet (select cohort or upload CSV)";
        }
        if (DOM.pipelineSeedInput && !DOM.pipelineSeedInput.value) {
            DOM.pipelineSeedInput.value = AppState.pipeline.seed;
        }
        if (DOM.pipelineEvalSeedInput && !DOM.pipelineEvalSeedInput.value) {
            DOM.pipelineEvalSeedInput.value = AppState.pipeline.evalSeed;
        }
        if (AppState.pipeline.trainRows) {
            renderPipelineStep1Results();
        }
    }

    function goToPipelineSubstep(substepNum) {
        substepNum = parseInt(substepNum, 10);
        if (substepNum > 1 && !AppState.pipeline.trainRows) {
            alert("Please execute Step 1: Train/Test Split & Strict Imputation first.");
            return;
        }
        if (substepNum === 7 && (!AppState.pipeline.rfImportance || AppState.pipeline.rfImportance.length === 0)) {
            alert("Please execute Step 6: Random Forest Scout & Feature Ranking first.");
            return;
        }

        AppState.pipeline.substep = substepNum;

        if (DOM.pipelineSubPills) {
            DOM.pipelineSubPills.forEach(pill => {
                const s = parseInt(pill.dataset.substep, 10);
                pill.classList.toggle('active', s === substepNum);
                if (s === substepNum) pill.setAttribute('aria-current', 'step'); else pill.removeAttribute('aria-current');
                if (s < substepNum) pill.classList.add('completed');
            });
        }

        if (DOM.pipelinePanels) {
            DOM.pipelinePanels.forEach(panel => {
                const isActive = panel.id === `pipeline-panel-${substepNum}`;
                panel.classList.toggle('active', isActive);
                panel.style.display = isActive ? 'block' : 'none';
            });
        }

        window.scrollTo({ top: 0, behavior: 'smooth' });

        if (substepNum === 2) renderPipelineStep2();
        if (substepNum === 3) renderPipelineStep3();
        if (substepNum === 4) renderPipelineStep4();
        if (substepNum === 5) renderPipelineStep5();
        if (substepNum === 6) renderPipelineStep6UI();
    }

    // =========================================================================
    // STEP 1: Train/Test Split (80/20) & Strict Leak-Free Imputation
    // =========================================================================
    function executePipelineStep1() {
        if (!AppState.parsedData || !AppState.parsedData.rows || AppState.parsedData.rows.length === 0) {
            alert("Please upload a CSV dataset or select a sample medical cohort first.");
            return;
        }

        // Identify target column
        let targetCol = Object.keys(AppState.columnConfigs).find(col => AppState.columnConfigs[col].role === 'target');
        if (!targetCol && AppState.parsedData.headers.length > 0) {
            targetCol = AppState.parsedData.headers[AppState.parsedData.headers.length - 1];
            if (AppState.columnConfigs[targetCol]) AppState.columnConfigs[targetCol].role = 'target';
        }

        if (!targetCol) {
            alert("Please select a target variable for prediction.");
            return;
        }

        // Identify feature columns
        const featureCols = Object.keys(AppState.columnConfigs).filter(col =>
            col !== targetCol && AppState.columnConfigs[col].role !== 'meta'
        );

        if (featureCols.length === 0) {
            alert("No predictor feature columns found. Ensure at least one column is configured as a feature.");
            return;
        }

        // Read configuration
        const userSeed = DOM.pipelineSeedInput ? parseSeed(DOM.pipelineSeedInput.value) : 42;
        AppState.pipeline.seed = userSeed === null ? 42 : userSeed;
        if (DOM.pipelineSeedInput) DOM.pipelineSeedInput.value = AppState.pipeline.seed; // show the seed actually used
        AppState.pipeline.evalSeed = AppState.pipeline.seed;
        if (DOM.pipelineEvalSeedInput) DOM.pipelineEvalSeedInput.value = AppState.pipeline.seed;

        const imputeStrategy = DOM.pipelineImputeStrategy ? DOM.pipelineImputeStrategy.value : 'median';
        AppState.pipeline.imputeStrategy = imputeStrategy;

        // Rows with a missing (or, for regression, non-numeric) outcome are excluded: outcomes are never imputed
        const isClassification = AppState.columnConfigs[targetCol]?.type !== 'continuous';
        const usableRows = AppState.parsedData.rows.filter(r => !DataEngine.isMissing(r[targetCol]) &&
            (isClassification || !isNaN(Number(r[targetCol]))));
        const excludedMissingTarget = AppState.parsedData.rows.length - usableRows.length;
        if (usableRows.length < 5) {
            alert(`Only ${usableRows.length} row(s) have a valid outcome for '${targetCol}'. At least 5 are needed for a train/test split.`);
            return;
        }

        // Train/test split at the selected ratio (default 80/20; stratified by outcome class for classification)
        const ratioEl = document.getElementById('pipeline-split-ratio');
        const pipeTrainRatio = ratioEl && !isNaN(parseFloat(ratioEl.value)) ? Math.min(0.9, Math.max(0.6, parseFloat(ratioEl.value))) : 0.8;
        AppState.pipeline.trainRatio = pipeTrainRatio;
        const targetVals = usableRows.map(r => String(r[targetCol]).trim());

        const splitResult = DataEngine.trainTestSplit(
            usableRows,
            targetVals,
            pipeTrainRatio,
            isClassification,
            AppState.pipeline.seed
        );

        const trainRaw = splitResult.trainX;
        const testRaw = splitResult.testX;

        // Fit strict imputer ONLY on the Training set (Zero Data Leakage)
        const imputer = DataEngine.fitStrictImputer(
            trainRaw,
            featureCols,
            targetCol,
            AppState.columnConfigs,
            imputeStrategy
        );

        // Transform both Training and Testing sets with the fitted imputer
        const trainTransformed = DataEngine.transformWithStrictImputer(trainRaw, imputer, AppState.columnConfigs);
        const testTransformed = DataEngine.transformWithStrictImputer(testRaw, imputer, AppState.columnConfigs);

        // Save pipeline state
        AppState.pipeline.trainRows = trainTransformed.rows;
        AppState.pipeline.testRows = testTransformed.rows;
        AppState.pipeline.imputer = imputer;
        AppState.pipeline.featureCols = featureCols;
        AppState.pipeline.targetCol = targetCol;
        AppState.pipeline.columnConfigs = { ...AppState.columnConfigs };
        AppState.pipeline.excludedMissingTarget = excludedMissingTarget;
        // Re-running Step 1 invalidates every downstream result
        AppState.pipeline.droppedFeatures = new Set();
        AppState.pipeline.normalityMap = {};
        AppState.pipeline.corrPairs = [];
        AppState.pipeline.vifResults = [];
        AppState.pipeline.univariateResults = [];
        AppState.pipeline.smotencResult = null;
        AppState.pipeline.balancedTrainRows = null;
        AppState.pipeline.rfImportance = [];
        AppState.pipeline.top70Features = [];
        AppState.pipeline.bottom30Features = [];
        // Old pipeline models are discarded by a new Step 1 (the deployed one is kept until replaced)
        disposeModels((AppState.pipeline.benchmarkResults || []).map(b => b.model));
        AppState.pipeline.benchmarkResults = [];
        if (DOM.pipelineResultsContainer) DOM.pipelineResultsContainer.style.display = 'none';
        if (DOM.pipelineRfImportanceContainer) DOM.pipelineRfImportanceContainer.innerHTML = '';
        if (DOM.pipelineToStep7Btn) DOM.pipelineToStep7Btn.disabled = true;
        if (DOM.pipelineVsBaselinePanel) DOM.pipelineVsBaselinePanel.style.display = 'none';
        if (AppState.deployedSource === '7step') {
            AppState.trainedModel = null;
            AppState.prepMeta = null;
            AppState.evalResults = null;
            AppState.deployedSource = null;
            AppState.deployedPipelineModel = null;
            updatePipelineActiveDeploymentStatus();
            updatePredictionProvenanceBar();
        }

        renderPipelineStep1Results(trainTransformed.missingFilledCount, testTransformed.missingFilledCount);
        if (excludedMissingTarget > 0) {
            alert(`${excludedMissingTarget} row(s) with a missing outcome ('${targetCol}') were excluded before splitting. Outcomes are never imputed.`);
        }
    }

    function renderPipelineStep1Results(trainMissing = 0, testMissing = 0) {
        if (!AppState.pipeline.trainRows || !AppState.pipeline.imputer) return;

        const trainRows = AppState.pipeline.trainRows;
        const testRows = AppState.pipeline.testRows;
        const imputer = AppState.pipeline.imputer;

        if (DOM.pipelineTrainCount) DOM.pipelineTrainCount.innerText = trainRows.length;
        if (DOM.pipelineTestCount) DOM.pipelineTestCount.innerText = testRows.length;
        if (DOM.pipelineMissingImputedCount) DOM.pipelineMissingImputedCount.innerText = trainMissing + testMissing;

        // Render imputer audit table
        let tableHtml = `
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Variable</th>
                        <th>Role</th>
                        <th>Measurement Scale</th>
                        <th>Fitted Imputation Value (Strictly from Train)</th>
                        <th>Training Summary Stats</th>
                    </tr>
                </thead>
                <tbody>
        `;

        Object.keys(imputer.fillValues).forEach(col => {
            const conf = AppState.pipeline.columnConfigs[col] || { type: 'continuous', role: 'feature' };
            const stats = imputer.columnStats[col] || {};
            const fillVal = imputer.fillValues[col];

            let statsStr = '';
            if (conf.type === 'nominal' || stats.mode !== undefined) {
                statsStr = `Mode: ${escapeHtml(stats.mode || 'N/A')} (Distinct: ${stats.distinctCount || 0})`;
            } else {
                statsStr = `Mean: ${stats.mean !== undefined ? stats.mean : 'N/A'}, Median: ${stats.median !== undefined ? stats.median : 'N/A'}, SD (sample, n−1): ${stats.std !== undefined ? stats.std : 'N/A'}`;
            }

            tableHtml += `
                <tr>
                    <td><strong>${escapeHtml(col)}</strong></td>
                    <td><span class="pill pill-${conf.role || 'feature'}">${conf.role || 'feature'}</span></td>
                    <td><span class="pill pill-${conf.type || 'continuous'}">${conf.type || 'continuous'}</span></td>
                    <td><strong style="color: var(--accent-emerald);">${escapeHtml(fillVal)}</strong></td>
                    <td style="font-size: 12px; color: var(--text-secondary);">${statsStr}</td>
                </tr>
            `;
        });

        tableHtml += `</tbody></table>`;
        tableHtml += `<p style="font-size: 12px; color: var(--text-secondary); margin-top: 8px;">
            Outcome '<strong>${escapeHtml(AppState.pipeline.targetCol)}</strong>' is never imputed${AppState.pipeline.excludedMissingTarget > 0
                ? `: ${AppState.pipeline.excludedMissingTarget} row(s) with a missing outcome were excluded before the split.`
                : '; no rows had a missing outcome.'}
        </p>`;

        if (DOM.pipelineImputerTableContainer) {
            DOM.pipelineImputerTableContainer.innerHTML = tableHtml;
        }

        if (DOM.pipelineStep1Results) {
            DOM.pipelineStep1Results.style.display = 'block';
        }
    }

    // =========================================================================
    // STEP 2: Continuous Variable Correlation Check (Pearson vs. Spearman)
    // =========================================================================
    function renderPipelineStep2() {
        if (!AppState.pipeline.trainRows) return;

        const contCols = AppState.pipeline.featureCols.filter(col =>
            !AppState.pipeline.droppedFeatures.has(col) &&
            (AppState.pipeline.columnConfigs[col]?.type === 'continuous' || AppState.pipeline.columnConfigs[col]?.type === 'ordinal')
        );

        if (contCols.length < 2) {
            if (DOM.pipelineNormalityTableContainer) {
                DOM.pipelineNormalityTableContainer.innerHTML = `<p style="padding: 16px; color: var(--text-muted);">Less than two continuous features available for pairwise correlation analysis.</p>`;
            }
            if (DOM.pipelineCorrPairsContainer) {
                DOM.pipelineCorrPairsContainer.innerHTML = `<p style="padding: 16px; color: var(--text-muted);">No continuous feature pairs to evaluate.</p>`;
            }
            if (DOM.pipelineCorrPairCounter) DOM.pipelineCorrPairCounter.innerText = `0 High Pairs`;
            return;
        }

        const corrRes = DataEngine.checkPairwiseCorrelations(AppState.pipeline.trainRows, contCols, 0.8, AppState.pipeline.columnConfigs);
        AppState.pipeline.normalityMap = corrRes.normalityMap;
        AppState.pipeline.corrPairs = corrRes.allPairs;

        // 1. Render Normality Table
        let normHtml = `
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Continuous Feature</th>
                        <th>Sample Mean</th>
                        <th>Std Deviation</th>
                        <th>Skewness (adjusted Fisher–Pearson G₁)</th>
                        <th>Skewness Screen (rule of thumb)</th>
                        <th>Correlation Method</th>
                    </tr>
                </thead>
                <tbody>
        `;

        contCols.forEach(col => {
            const norm = corrRes.normalityMap[col] || { mean: 0, std: 1, skewness: 0, isNormal: true };
            normHtml += `
                <tr>
                    <td><strong>${escapeHtml(col)}</strong></td>
                    <td>${norm.mean}</td>
                    <td>${norm.std}</td>
                    <td><strong style="color: ${norm.isNormal ? 'var(--accent-emerald)' : 'var(--accent-amber)'};">${norm.skewness}</strong></td>
                    <td>
                        <span class="pill" style="${norm.isNormal ? 'background: rgba(16, 185, 129, 0.15); color: var(--accent-emerald); border-color: rgba(16, 185, 129, 0.4);' : 'background: rgba(245, 158, 11, 0.15); color: var(--accent-amber); border-color: rgba(245, 158, 11, 0.4);'}">
                            ${norm.isNormal ? '✓ Approx. symmetric (|G₁| ≤ 1.0)' : '⚠️ Skewed (|G₁| > 1.0)'}
                        </span>
                    </td>
                    <td style="font-size: 12px; color: var(--text-secondary);">
                        ${norm.isNormal ? 'Pearson if its partner is also approx. symmetric' : 'Spearman rank correlation for its pairs'}
                    </td>
                </tr>
            `;
        });
        normHtml += `</tbody></table>`;
        if (DOM.pipelineNormalityTableContainer) DOM.pipelineNormalityTableContainer.innerHTML = normHtml;

        // 2. Render High Correlation Pairs (|r| or |rho| >= 0.8)
        const highPairs = corrRes.highPairs;
        if (DOM.pipelineCorrPairCounter) {
            DOM.pipelineCorrPairCounter.innerText = `${highPairs.length} High Pairs`;
        }

        if (highPairs.length === 0) {
            DOM.pipelineCorrPairsContainer.innerHTML = `
                <div class="pipeline-alert pipeline-alert-success">
                    <div style="font-size: 18px;">✅</div>
                    <div>
                        <strong>No Excessive Pairwise Collinearity Detected:</strong> All continuous variable pairs have |correlation| &lt; 0.80. No automatic dropping is strictly required; you may proceed to Step 3.
                    </div>
                </div>
            `;
        } else {
            let pairsHtml = '';
            highPairs.forEach((p, idx) => {
                const var1Dropped = AppState.pipeline.droppedFeatures.has(p.var1);
                const var2Dropped = AppState.pipeline.droppedFeatures.has(p.var2);

                pairsHtml += `
                    <div class="corr-pair-card high-corr ${var1Dropped || var2Dropped ? 'dropped' : ''}">
                        <div>
                            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                                <strong style="font-size: 14px; color: var(--text-primary);">${escapeHtml(p.var1)}</strong>
                                <span style="color: var(--text-muted);">&harr;</span>
                                <strong style="font-size: 14px; color: var(--text-primary);">${escapeHtml(p.var2)}</strong>
                                <span class="vif-badge-danger">${p.method} = ${p.coeff}</span>
                            </div>
                            <div style="font-size: 12px; color: var(--text-secondary);">
                                <strong>Method:</strong> ${p.bothNormal ? 'Both variables approximately symmetric (|G₁| ≤ 1) &rarr; Pearson product-moment correlation.' : `Skewness detected (${escapeHtml([p.norm1.isNormal ? '' : p.var1, p.norm2.isNormal ? '' : p.var2].filter(Boolean).join(', '))}) &rarr; Spearman rank correlation.`}
                            </div>
                        </div>
                        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                            <button class="btn btn-rose btn-sm" onclick="window.pipelineDropFeature(${jsArg(p.var1)})" ${var1Dropped ? 'disabled' : ''}>
                                ${var1Dropped ? `✓ Dropped ${escapeHtml(p.var1)}` : `🗑️ Drop ${escapeHtml(p.var1)}`}
                            </button>
                            <button class="btn btn-rose btn-sm" onclick="window.pipelineDropFeature(${jsArg(p.var2)})" ${var2Dropped ? 'disabled' : ''}>
                                ${var2Dropped ? `✓ Dropped ${escapeHtml(p.var2)}` : `🗑️ Drop ${escapeHtml(p.var2)}`}
                            </button>
                            <button class="btn btn-secondary btn-sm" onclick="window.pipelineKeepBothPairs(${jsArg(p.var1)}, ${jsArg(p.var2)})">
                                🤝 Keep Both
                            </button>
                        </div>
                    </div>
                `;
            });
            DOM.pipelineCorrPairsContainer.innerHTML = pairsHtml;
        }
    }

    window.pipelineDropFeature = function (featureName) {
        AppState.pipeline.droppedFeatures.add(featureName);
        renderPipelineStep2();
        if (AppState.pipeline.substep === 3) renderPipelineStep3();
    };

    window.pipelineKeepBothPairs = function (var1, var2) {
        alert(`Pair [${var1}] and [${var2}] retained. Record the clinical justification for keeping both in your study protocol.`);
    };

    // =========================================================================
    // STEP 3: Variance Inflation Factor (VIF) Check
    // =========================================================================
    function renderPipelineStep3() {
        if (!AppState.pipeline.trainRows) return;

        const activeCols = AppState.pipeline.featureCols.filter(c => !AppState.pipeline.droppedFeatures.has(c));
        const vifResults = DataEngine.calculateVIF(AppState.pipeline.trainRows, activeCols, AppState.pipeline.columnConfigs);
        AppState.pipeline.vifResults = vifResults;

        const hasHighVif = vifResults.some(v => v.highVIF);
        if (DOM.pipelineDropVifBtn) DOM.pipelineDropVifBtn.disabled = !hasHighVif;

        let tableHtml = `
            <table class="data-table">
                <thead>
                    <tr>
                        <th style="width: 40px; text-align: center;">Select</th>
                        <th>Feature Name</th>
                        <th>Measurement Scale</th>
                        <th>df (Encoded Columns)</th>
                        <th>Equivalent R²</th>
                        <th>VIF (GVIF<sup>1/df</sup> for df &gt; 1)</th>
                        <th>Collinearity Status</th>
                    </tr>
                </thead>
                <tbody>
        `;

        vifResults.forEach(item => {
            const isHigh = item.highVIF;
            const isDropped = AppState.pipeline.droppedFeatures.has(item.feature);

            tableHtml += `
                <tr style="${isHigh ? 'background: rgba(244, 63, 94, 0.04);' : ''}">
                    <td style="text-align: center;">
                        <input type="checkbox" class="pipeline-vif-checkbox" data-feature="${escapeHtml(item.feature)}" ${isHigh && !isDropped ? 'checked' : ''} ${isDropped ? 'disabled' : ''}>
                    </td>
                    <td><strong>${escapeHtml(item.feature)}</strong> ${isDropped ? '<span class="pill pill-meta" style="font-size: 10px;">Dropped</span>' : ''}</td>
                    <td><span class="pill pill-${item.type}">${item.type}</span></td>
                    <td style="font-size: 12px; color: var(--text-secondary);">${item.dummyCount}</td>
                    <td>${item.rSquared}</td>
                    <td><strong style="color: ${isHigh ? 'var(--accent-rose)' : 'var(--accent-emerald)'}; font-size: 13.5px;">${item.vif >= 999 ? '≥ 999 (≈ perfect collinearity)' : item.vif}</strong>${item.dummyCount > 1 && item.vif < 999 ? `<div style="font-size: 10.5px; color: var(--text-muted);">GVIF ${item.gvif}; GVIF<sup>1/(2·df)</sup> ${item.adjustedGVIF}</div>` : ''}</td>
                    <td>
                        <span class="${isHigh ? 'vif-badge-danger' : 'vif-badge-ok'}">
                            ${isHigh ? '⚠️ VIF > 5.0 (High Multicollinearity)' : '✓ VIF ≤ 5.0 (Acceptable)'}
                        </span>
                    </td>
                </tr>
            `;
        });

        tableHtml += `</tbody></table>`;
        if (DOM.pipelineVifTableContainer) DOM.pipelineVifTableContainer.innerHTML = tableHtml;
    }

    function pipelineDropSelectedVIF() {
        const checkboxes = document.querySelectorAll('.pipeline-vif-checkbox:checked');
        if (checkboxes.length === 0) {
            alert("No variables selected to eliminate.");
            return;
        }

        const droppedNames = [];
        checkboxes.forEach(cb => {
            const f = cb.dataset.feature;
            if (f) {
                AppState.pipeline.droppedFeatures.add(f);
                droppedNames.push(f);
            }
        });

        alert(`Excluded ${droppedNames.length} feature(s) from the pipeline (Train and Test): ${droppedNames.join(', ')}. They can be restored in Step 4.`);
        renderPipelineStep3();
    }

    // =========================================================================
    // STEP 4: Univariate Statistical Testing Check
    // =========================================================================
    function renderPipelineStep4() {
        if (!AppState.pipeline.trainRows) return;

        // Run univariate test on all pipeline features so user can see rankings and restore any previously dropped variables
        const allFeatures = AppState.pipeline.featureCols;
        const results = DataEngine.runUnivariateTesting(
            AppState.pipeline.trainRows,
            allFeatures,
            AppState.pipeline.targetCol,
            AppState.pipeline.columnConfigs
        );

        AppState.pipeline.univariateResults = results;

        const activeCount = allFeatures.filter(c => !AppState.pipeline.droppedFeatures.has(c)).length;
        const sigCount = results.filter(r => r.significant && !AppState.pipeline.droppedFeatures.has(r.feature)).length;
        const nonSigCount = activeCount - sigCount;

        if (DOM.pipelineUnivSelectedSummary) {
            DOM.pipelineUnivSelectedSummary.innerHTML = `
                Active in Pipeline: <strong>${activeCount} / ${allFeatures.length}</strong> features 
                (${sigCount} significant, ${nonSigCount} non-significant). 
                ${AppState.pipeline.droppedFeatures.size > 0 ? `<span style="color: var(--accent-rose);">(${AppState.pipeline.droppedFeatures.size} currently excluded)</span>` : ''}
            `;
        }

        let tableHtml = `
            <table class="data-table">
                <thead>
                    <tr>
                        <th style="width: 40px; text-align: center;">
                            <input type="checkbox" id="pipeline-univ-master-check" title="Toggle all active" checked>
                        </th>
                        <th style="width: 50px;">Rank</th>
                        <th>Feature</th>
                        <th>Measurement Scale</th>
                        <th>Statistical Test</th>
                        <th>Test Statistic</th>
                        <th>Degrees of Freedom</th>
                        <th>p-value</th>
                        <th>Significance</th>
                        <th style="text-align: right; width: 110px;">Action</th>
                    </tr>
                </thead>
                <tbody>
        `;

        results.forEach((r, idx) => {
            const isSig = r.significant;
            const isDropped = AppState.pipeline.droppedFeatures.has(r.feature);

            tableHtml += `
                <tr style="${isDropped ? 'opacity: 0.55; background: rgba(244, 63, 94, 0.04);' : (isSig ? 'background: rgba(16, 185, 129, 0.02);' : '')}">
                    <td style="text-align: center;">
                        <input type="checkbox" class="pipeline-univ-checkbox" data-feature="${escapeHtml(r.feature)}" data-sig="${isSig}" ${!isDropped ? 'checked' : ''}>
                    </td>
                    <td style="font-weight: 700; color: var(--accent-cyan);">#${idx + 1}</td>
                    <td>
                        <strong style="${isDropped ? 'text-decoration: line-through;' : ''}">${escapeHtml(r.feature)}</strong>
                        ${isDropped ? '<span class="pill pill-meta" style="font-size: 10px; margin-left: 6px;">Excluded</span>' : ''}
                    </td>
                    <td><span class="pill pill-${r.scale.toLowerCase().includes('nominal') ? 'nominal' : 'continuous'}">${r.scale}</span></td>
                    <td style="font-weight: 600;">${escapeHtml(r.testName)}</td>
                    <td>${escapeHtml(r.statisticLabel)}: <strong>${r.statisticValue === Infinity ? '∞' : r.statisticValue}</strong></td>
                    <td style="font-size: 12px; color: var(--text-secondary);">${r.df}</td>
                    <td><strong style="color: ${isSig ? 'var(--accent-emerald)' : 'var(--text-muted)'}; font-size: 13.5px;">${r.pValue < 0.0001 ? '< 0.0001' : r.pValue}</strong></td>
                    <td>
                        <span class="pill" style="${isSig ? 'background: rgba(16, 185, 129, 0.15); color: var(--accent-emerald); border-color: rgba(16, 185, 129, 0.4);' : 'background: rgba(100, 116, 139, 0.15); color: var(--text-muted);'}">
                            ${isSig ? '✓ p < 0.05 (Sig)' : 'p ≥ 0.05 (Non-Sig)'}
                        </span>
                    </td>
                    <td style="text-align: right;">
                        ${isDropped ? `
                            <button class="btn btn-secondary btn-sm" onclick="window.pipelineRestoreFeatureStep4(${jsArg(r.feature)})" title="Restore variable into modeling pipeline">
                                ↩️ Include
                            </button>
                        ` : `
                            <button class="btn btn-rose btn-sm" onclick="window.pipelineDropFeatureStep4(${jsArg(r.feature)})" title="Exclude this variable from modeling">
                                🗑️ Exclude
                            </button>
                        `}
                    </td>
                </tr>
            `;
        });

        tableHtml += `</tbody></table>`;
        if (DOM.pipelineUnivariateTableContainer) {
            DOM.pipelineUnivariateTableContainer.innerHTML = tableHtml;

            // Master checkbox event
            const masterCheck = document.getElementById('pipeline-univ-master-check');
            if (masterCheck) {
                masterCheck.addEventListener('change', (e) => {
                    const cbs = document.querySelectorAll('.pipeline-univ-checkbox');
                    cbs.forEach(cb => cb.checked = e.target.checked);
                    updateUnivSelectionSummary();
                });
            }

            // Individual checkbox events
            const cbs = document.querySelectorAll('.pipeline-univ-checkbox');
            cbs.forEach(cb => {
                cb.addEventListener('change', updateUnivSelectionSummary);
            });
        }
    }

    function updateUnivSelectionSummary() {
        const cbs = document.querySelectorAll('.pipeline-univ-checkbox');
        const checkedCount = Array.from(cbs).filter(cb => cb.checked).length;
        if (DOM.pipelineUnivSelectedSummary) {
            DOM.pipelineUnivSelectedSummary.innerHTML = `
                Checked for Inclusion: <strong>${checkedCount} / ${cbs.length}</strong> features. 
                Click <strong>"🗑️ Exclude Unchecked Features"</strong> to apply changes to subsequent pipeline steps.
            `;
        }
    }

    function filterSignificantUnivariateFeatures() {
        const cbs = document.querySelectorAll('.pipeline-univ-checkbox');
        let uncheckCount = 0;
        cbs.forEach(cb => {
            const isSig = cb.dataset.sig === 'true';
            if (!isSig) {
                cb.checked = false;
                uncheckCount++;
            } else {
                cb.checked = true;
            }
        });
        updateUnivSelectionSummary();
        alert(`Unchecked ${uncheckCount} non-significant feature(s) (p ≥ 0.05). Click "🗑️ Exclude Unchecked Features" to finalize.`);
    }

    function selectAllUnivariateFeatures() {
        const cbs = document.querySelectorAll('.pipeline-univ-checkbox');
        cbs.forEach(cb => cb.checked = true);
        const masterCheck = document.getElementById('pipeline-univ-master-check');
        if (masterCheck) masterCheck.checked = true;
        updateUnivSelectionSummary();
    }

    function dropUnselectedUnivariateFeatures() {
        const cbs = document.querySelectorAll('.pipeline-univ-checkbox');
        let newlyDropped = [];
        let newlyRestored = [];

        cbs.forEach(cb => {
            const f = cb.dataset.feature;
            if (!f) return;
            if (!cb.checked) {
                if (!AppState.pipeline.droppedFeatures.has(f)) {
                    AppState.pipeline.droppedFeatures.add(f);
                    newlyDropped.push(f);
                }
            } else {
                if (AppState.pipeline.droppedFeatures.has(f)) {
                    AppState.pipeline.droppedFeatures.delete(f);
                    newlyRestored.push(f);
                }
            }
        });

        const activeCount = AppState.pipeline.featureCols.filter(c => !AppState.pipeline.droppedFeatures.has(c)).length;
        if (activeCount === 0) {
            alert("Warning: At least 1 predictor variable must remain active in the pipeline!");
            return;
        }

        let msg = `Pipeline features updated successfully! (${activeCount} active)`;
        if (newlyDropped.length > 0) msg += `\nExcluded: ${newlyDropped.join(', ')}`;
        if (newlyRestored.length > 0) msg += `\nRestored: ${newlyRestored.join(', ')}`;
        alert(msg);

        renderPipelineStep4();
    }

    window.pipelineDropFeatureStep4 = function (featureName) {
        AppState.pipeline.droppedFeatures.add(featureName);
        renderPipelineStep4();
    };

    window.pipelineRestoreFeatureStep4 = function (featureName) {
        AppState.pipeline.droppedFeatures.delete(featureName);
        renderPipelineStep4();
    };

    // =========================================================================
    // STEP 5: Imbalance Handling (SMOTENC)
    // =========================================================================
    function renderPipelineStep5() {
        if (!AppState.pipeline.trainRows) return;

        const activeCols = AppState.pipeline.featureCols.filter(c => !AppState.pipeline.droppedFeatures.has(c));
        const smotencRes = DataEngine.applySMOTENC(
            AppState.pipeline.trainRows,
            activeCols,
            AppState.pipeline.targetCol,
            AppState.pipeline.columnConfigs,
            AppState.pipeline.seed
        );

        AppState.pipeline.smotencResult = smotencRes;
        AppState.pipeline.balancedTrainRows = smotencRes.balancedRows;

        let contentHtml = '';
        if (smotencRes.applied) {
            contentHtml = `
                <div class="pipeline-alert pipeline-alert-warning" style="margin-bottom: 20px;">
                    <div style="font-size: 20px;">⚠️</div>
                    <div>
                        <strong>Class Imbalance Detected:</strong><br>
                        ${escapeHtml(smotencRes.reason)}<br>
                        Smallest class <code>${escapeHtml(smotencRes.minorityClass)}</code>: ${smotencRes.minorityPctBefore}% of training rows before &rarr; ${smotencRes.minorityPctAfter}% after resampling.
                        Synthetic rows are used for training only; all test-set metrics are computed on real patients at their true prevalence.
                    </div>
                </div>

                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 16px;">
                    <div class="card" style="padding: 16px; background: var(--bg-surface-subtle); border: 1px solid var(--border-color);">
                        <h5 style="font-size: 13px; font-weight: 700; margin-bottom: 10px; color: var(--text-secondary);">Original Training Class Distribution</h5>
                        ${Object.keys(smotencRes.originalCounts).map(c => `
                            <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 12.5px;">
                                <span>Class <strong>${escapeHtml(c)}</strong>:</span>
                                <strong>${smotencRes.originalCounts[c]} cases</strong>
                            </div>
                        `).join('')}
                    </div>
                    <div class="card" style="padding: 16px; background: rgba(16, 185, 129, 0.05); border: 1px solid rgba(16, 185, 129, 0.3);">
                        <h5 style="font-size: 13px; font-weight: 700; margin-bottom: 10px; color: var(--accent-emerald);">Post-SMOTENC Training Class Distribution</h5>
                        ${Object.keys(smotencRes.balancedCounts).map(c => `
                            <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 12.5px;">
                                <span>Class <strong>${escapeHtml(c)}</strong>:</span>
                                <strong style="color: var(--accent-emerald);">${smotencRes.balancedCounts[c]} cases</strong>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        } else {
            contentHtml = `
                <div class="pipeline-alert pipeline-alert-success" style="margin-bottom: 20px;">
                    <div style="font-size: 20px;">✅</div>
                    <div>
                        <strong>No Synthetic Resampling Applied:</strong><br>
                        ${escapeHtml(smotencRes.reason || 'No synthetic resampling needed.')}
                    </div>
                </div>
                <div class="card" style="padding: 16px; background: var(--bg-surface-subtle); border: 1px solid var(--border-color);">
                    <h5 style="font-size: 13px; font-weight: 700; margin-bottom: 10px; color: var(--text-secondary);">Training Class Distribution</h5>
                    ${Object.keys(smotencRes.classCounts || {}).map(c => `
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 12.5px;">
                            <span>Class <strong>${escapeHtml(c)}</strong>:</span>
                            <strong>${smotencRes.classCounts[c]} cases</strong>
                        </div>
                    `).join('')}
                </div>
            `;
        }

        if (DOM.pipelineSmotencContainer) {
            DOM.pipelineSmotencContainer.innerHTML = contentHtml;
        }
    }

    // =========================================================================
    // STEP 6: Model-Driven Feature Ranking (Random Forest Scout)
    // =========================================================================
    function renderPipelineStep6UI() {
        if (!AppState.pipeline.rfImportance || AppState.pipeline.rfImportance.length === 0) {
            if (DOM.pipelineRfImportanceContainer) {
                DOM.pipelineRfImportanceContainer.innerHTML = `
                    <p style="padding: 24px; color: var(--text-muted); text-align: center;">
                        Click the purple 'Run Random Forest Scout & Feature Ranking' button above to train a Random Forest classifier on the balanced training set and extract feature importances.
                    </p>
                `;
            }
        }
    }

    function executePipelineStep6() {
        // Always recompute Step 5 from the current split & feature set so the scout never uses stale rows
        renderPipelineStep5();

        const activeCols = AppState.pipeline.featureCols.filter(c => !AppState.pipeline.droppedFeatures.has(c));
        if (activeCols.length === 0) {
            alert("No active predictor features remain. Restore at least one feature in Step 4.");
            return;
        }
        const subConfigs = {};
        activeCols.forEach(c => subConfigs[c] = AppState.pipeline.columnConfigs[c]);
        subConfigs[AppState.pipeline.targetCol] = AppState.pipeline.columnConfigs[AppState.pipeline.targetCol];

        const prep = DataEngine.prepareDataset(
            AppState.pipeline.balancedTrainRows,
            subConfigs,
            activeCols,
            AppState.pipeline.targetCol
        );

        // Train Random Forest Scout
        const rf = new RandomForest(prep.isClassification, 40, 6, 2, 'sqrt', AppState.pipeline.seed);
        rf.fit(prep.X, prep.y);

        // Map feature importances back to original clinical features
        const featureImpMap = {};
        activeCols.forEach(col => {
            const meta = prep.featureMetadata[col];
            let sumImp = 0;
            if (meta) {
                for (let k = 0; k < meta.length; k++) {
                    sumImp += rf.featureImportances[meta.startIndex + k] || 0;
                }
            }
            featureImpMap[col] = sumImp;
        });

        const totalImp = Object.values(featureImpMap).reduce((a, b) => a + b, 0) || 1;
        const ranked = activeCols.map(col => ({
            name: col,
            importance: Number((featureImpMap[col] / totalImp).toFixed(4)),
            pct: Number(((featureImpMap[col] / totalImp) * 100).toFixed(1))
        }));

        ranked.sort((a, b) => b.importance - a.importance);

        // Bottom 30% cutoff
        const k = Math.max(1, Math.round(activeCols.length * 0.30));
        const topCount = activeCols.length - k;

        AppState.pipeline.top70Features = ranked.slice(0, topCount).map(f => f.name);
        AppState.pipeline.bottom30Features = ranked.slice(topCount).map(f => f.name);
        AppState.pipeline.rfImportance = ranked;

        // Render ranked bar chart
        const maxPct = Math.max(...ranked.map(r => r.pct), 1);
        let chartHtml = `
            <div class="rf-importance-bar-container">
        `;

        ranked.forEach((r, idx) => {
            const isBottomTier = idx >= topCount;
            const barWidth = Math.max(4, Math.round((r.pct / maxPct) * 100));

            if (idx === topCount) {
                chartHtml += `
                    <div class="cutoff-line-notice">
                        <span>✂️</span>
                        <span><strong>Bottom 30% Cutoff:</strong> Features below this line are removed for k-NN and Naive Bayes only in Step 7 (other models use all surviving features). MDI importance is computed on the SMOTENC-balanced training set and can favour features with many distinct values.</span>
                    </div>
                `;
            }

            chartHtml += `
                <div class="rf-importance-row">
                    <div>
                        <strong style="color: ${isBottomTier ? 'var(--accent-amber)' : 'var(--text-primary)'};">#${idx + 1} ${escapeHtml(r.name)}</strong>
                    </div>
                    <div class="rf-meter-track">
                        <div class="rf-meter-fill ${isBottomTier ? 'bottom-tier' : 'top-tier'}" style="width: ${barWidth}%;"></div>
                    </div>
                    <div style="font-weight: 700; color: ${isBottomTier ? 'var(--accent-amber)' : 'var(--accent-emerald)'}; text-align: right;">
                        ${r.pct}%
                    </div>
                    <div>
                        <span class="pill" style="font-size: 10px; ${isBottomTier ? 'background: rgba(245,158,11,0.15); color: var(--accent-amber); border-color: rgba(245,158,11,0.4);' : 'background: rgba(16,185,129,0.15); color: var(--accent-emerald); border-color: rgba(16,185,129,0.4);'}">
                            ${isBottomTier ? 'Bottom 30%' : 'Top 70%'}
                        </span>
                    </div>
                </div>
            `;
        });

        chartHtml += `</div>`;
        if (DOM.pipelineRfImportanceContainer) DOM.pipelineRfImportanceContainer.innerHTML = chartHtml;
        if (DOM.pipelineToStep7Btn) DOM.pipelineToStep7Btn.disabled = false;
    }

    // =========================================================================
    // STEP 7: Algorithm-Specific Training & Constraints
    // =========================================================================
    async function executePipelineStep7() {
        if (!AppState.pipeline.rfImportance || AppState.pipeline.rfImportance.length === 0) {
            alert("Please complete Step 6: Random Forest Scout & Feature Ranking first.");
            return;
        }

        const userSeed = DOM.pipelineEvalSeedInput ? parseSeed(DOM.pipelineEvalSeedInput.value) : 42;
        AppState.pipeline.evalSeed = userSeed === null ? 42 : userSeed;
        if (DOM.pipelineEvalSeedInput) DOM.pipelineEvalSeedInput.value = AppState.pipeline.evalSeed;

        if (DOM.pipelineTrainModelsBtn) DOM.pipelineTrainModelsBtn.disabled = true;
        if (DOM.pipelineResultsContainer) DOM.pipelineResultsContainer.style.display = 'none';

        const watchdog = AppState.watchdogs ? AppState.watchdogs.pipeline : null;
        if (watchdog) {
            watchdog.start("Initializing constrained clinical pipeline...", 5, (reason) => {
                console.warn("Pipeline Step 7 training stopped:", reason);
                if (DOM.pipelineTrainModelsBtn) DOM.pipelineTrainModelsBtn.disabled = false;
            });
        } else if (DOM.pipelineTrainStatus) {
            DOM.pipelineTrainStatus.style.display = 'block';
        }

        try {
            await new Promise(r => setTimeout(r, 60));

            // Recompute Step 5 so the balanced rows match the current split and feature set
            renderPipelineStep5();
            if (!AppState.pipeline.balancedTrainRows) {
                throw new Error("Training rows are unavailable. Please re-run Step 1.");
            }

            const activeCols = AppState.pipeline.featureCols.filter(c => !AppState.pipeline.droppedFeatures.has(c));
            if (activeCols.length === 0) {
                throw new Error("No active predictor features remain. Restore at least one feature in Step 4.");
            }
            
            // Top 70% features with safe fallback
            let top70Cols = AppState.pipeline.top70Features;
            if (!top70Cols || top70Cols.length === 0) {
                const k = Math.max(1, Math.round(activeCols.length * 0.30));
                const topCount = Math.max(1, activeCols.length - k);
                top70Cols = activeCols.slice(0, topCount);
            }
            top70Cols = top70Cols.filter(c => !AppState.pipeline.droppedFeatures.has(c));
            if (top70Cols.length === 0) top70Cols = [...activeCols];

            const targetCol = AppState.pipeline.targetCol;
            const isClassification = AppState.pipeline.columnConfigs[targetCol]?.type !== 'continuous';
            // Model plan: Naive Bayes is classification-only, so regression runs 4 models, not 5
            const plannedModels = isClassification ? ['knn', 'nb', 'rf', 'nn', 'glm', 'ensemble'] : ['knn', 'rf', 'nn', 'glm', 'ensemble'];
            const totalModels = plannedModels.length;
            const stepOf = id => plannedModels.indexOf(id) + 1;
            if (watchdog) watchdog.totalSteps = totalModels;

            // 1. Prepare standard full dataset (for DNN, GLM, Ensemble)
            const fullConfigs = {};
            activeCols.forEach(c => fullConfigs[c] = AppState.pipeline.columnConfigs[c]);
            fullConfigs[targetCol] = AppState.pipeline.columnConfigs[targetCol];

            const fullPrep = DataEngine.prepareDataset(
                AppState.pipeline.balancedTrainRows,
                fullConfigs,
                activeCols,
                targetCol
            );
            // Single source for the class count: the encoder's class list
            const numClasses = isClassification ? fullPrep.targetClasses.length : 1;

            // Transform unaltered Test set using full training metadata
            const fullTestTrans = DataEngine.prepareTestingDataset(AppState.pipeline.testRows, fullPrep);
            // Exploratory 2×2 cutoff for continuous outcomes: median of the (real, non-synthetic) training outcomes
            const pipeEvalOptions = {
                regressionThreshold: isClassification ? null : trainingMedian(AppState.pipeline.trainRows.map(r => Number(r[targetCol]))),
                thresholdSource: 'median of the training-split outcomes'
            };

            // 2. Prepare reduced Top 70% dataset (strictly for k-NN & Naive Bayes)
            const top70Configs = {};
            top70Cols.forEach(c => top70Configs[c] = AppState.pipeline.columnConfigs[c]);
            top70Configs[targetCol] = AppState.pipeline.columnConfigs[targetCol];

            const top70Prep = DataEngine.prepareDataset(
                AppState.pipeline.balancedTrainRows,
                top70Configs,
                top70Cols,
                targetCol
            );
            const top70TestTrans = DataEngine.prepareTestingDataset(AppState.pipeline.testRows, top70Prep);

            const benchmarkList = [];

            // A. k-NN Model (Strict constraint: Bottom 30% features dropped)
            if (watchdog) watchdog.tick(stepOf('knn'), `Training Model ${stepOf('knn')}/${totalModels}: k-NN (Top 70% features only)...`);
            await new Promise(r => setTimeout(r, 60));
            const knn = new KNNModel(isClassification, 5, 'euclidean', 'distance', AppState.pipeline.evalSeed);
            knn.fit(top70Prep.X, top70Prep.y);
            const knnEval = ModelEngine.evaluate(knn, top70TestTrans.X, top70TestTrans.y, isClassification, fullPrep.targetClasses, pipeEvalOptions);
            knnEval.featuresUsed = `Top 70% (${top70Cols.length} features)`;
            knnEval.constraintNote = "Bottom 30% RF-importance features removed to reduce distance noise in high dimensions.";
            benchmarkList.push({ id: 'knn', name: 'k-Nearest Neighbors (k-NN)', model: knn, prep: top70Prep, eval: knnEval });

            // B. Gaussian Naive Bayes (Strict constraint: Bottom 30% features dropped)
            if (isClassification) {
                if (watchdog) watchdog.tick(stepOf('nb'), `Training Model ${stepOf('nb')}/${totalModels}: Naive Bayes (Top 70% features only)...`);
                await new Promise(r => setTimeout(r, 60));
                const nb = new GaussianNaiveBayes(1e-4, 'empirical', AppState.pipeline.evalSeed);
                nb.fit(top70Prep.X, top70Prep.y);
                const nbEval = ModelEngine.evaluate(nb, top70TestTrans.X, top70TestTrans.y, isClassification, fullPrep.targetClasses, pipeEvalOptions);
                nbEval.featuresUsed = `Top 70% (${top70Cols.length} features)`;
                nbEval.constraintNote = "Bottom 30% RF-importance features removed (fewer noisy inputs; does not make features conditionally independent).";
                benchmarkList.push({ id: 'nb', name: 'Naive Bayes (Gaussian + Bernoulli)', model: nb, prep: top70Prep, eval: nbEval });
            }

            // B2. Random Forest (all surviving features). Tree splits are scale-free and the forest
            // selects features per split, so no RF-importance pruning is applied.
            if (watchdog) watchdog.tick(stepOf('rf'), `Training Model ${stepOf('rf')}/${totalModels}: Random Forest (all surviving features)...`);
            await new Promise(r => setTimeout(r, 60));
            const rfFinal = new RandomForest(isClassification, 40, 6, 2, 'sqrt', AppState.pipeline.evalSeed);
            await fitWithYields(rfFinal, fullPrep.X, fullPrep.y, watchdog);
            const rfEval = ModelEngine.evaluate(rfFinal, fullTestTrans.X, fullTestTrans.y, isClassification, fullPrep.targetClasses, pipeEvalOptions);
            rfEval.featuresUsed = `All Surviving (${activeCols.length} features)`;
            rfEval.constraintNote = "40 trees, max depth 6, √p features per split; probabilities are tree vote shares.";
            benchmarkList.push({ id: 'rf', name: 'Random Forest', model: rfFinal, prep: fullPrep, eval: rfEval });

            // C. Deep Neural Network (Strict constraint: L2 Weight Decay + Dropout)
            if (watchdog) watchdog.tick(stepOf('nn'), `Training Model ${stepOf('nn')}/${totalModels}: Neural Network (MLP; L2 + dropout + early stopping)...`);
            await new Promise(r => setTimeout(r, 60));
            const dnnParams = {
                epochs: 25,
                learningRate: 0.005,
                hiddenLayers: [48, 24],
                dropoutRate: 0.15,
                l2Reg: 0.001,
                activation: 'relu',
                optimizer: 'adam',
                batchSize: 16,
                seed: AppState.pipeline.evalSeed
            };
            const dnn = new DeepNeuralNetwork(isClassification, numClasses, dnnParams);
            // The test split is never shown to the model: the NN carves its own validation split from the training rows
            await dnn.fit(fullPrep.X, fullPrep.y, null, null, (ep, tot) => {
                if (watchdog && watchdog.aborted) throw new Error("Training aborted by user."); // stop at this epoch
                if (watchdog && (ep % 4 === 0 || ep === tot)) {
                    watchdog.tick(stepOf('nn'), `Training Model ${stepOf('nn')}/${totalModels}: Neural Network (Epoch ${ep}/${tot})...`);
                }
            });
            const dnnEval = ModelEngine.evaluate(dnn, fullTestTrans.X, fullTestTrans.y, isClassification, fullPrep.targetClasses, pipeEvalOptions);
            dnnEval.featuresUsed = `All Surviving (${activeCols.length} features)`;
            dnnEval.constraintNote = dnn.usedFallback
                ? `⚠️ TensorFlow.js failed (${dnn.fallbackReason}); a linear fallback model was trained instead of a neural network.`
                : `L2 weight decay (0.001) and dropout (0.15); ${dnn.trainingInfo ? describeNNTraining(dnn.trainingInfo) : ''}`;
            benchmarkList.push({ id: 'nn', name: displayModelName('nn', dnn), model: dnn, prep: fullPrep, eval: dnnEval });

            // D. Logistic / Ridge Regression (Strict constraint: L2 penalization against collinearity)
            if (watchdog) watchdog.tick(stepOf('glm'), `Training Model ${stepOf('glm')}/${totalModels}: Logistic / Ridge Regression (L2 lambda = 0.02)...`);
            await new Promise(r => setTimeout(r, 60));
            const glm = new GeneralizedLinearModel(
                isClassification,
                numClasses,
                0.05,
                150,
                0.02,
                'momentum',
                AppState.pipeline.evalSeed
            );
            await fitWithYields(glm, fullPrep.X, fullPrep.y, watchdog);
            const glmEval = ModelEngine.evaluate(glm, fullTestTrans.X, fullTestTrans.y, isClassification, fullPrep.targetClasses, pipeEvalOptions);
            glmEval.featuresUsed = `All Surviving (${activeCols.length} features)`;
            glmEval.constraintNote = "Applied Ridge L2 penalization (λ = 0.02) to stabilize against minor collinearity.";
            benchmarkList.push({ id: 'glm', name: 'Logistic / Ridge Regression', model: glm, prep: fullPrep, eval: glmEval });

            // E. Ensemble Super Learner (Strict constraint: StandardScaler fitted on Train and applied to Test)
            if (watchdog) watchdog.tick(stepOf('ensemble'), `Training Model ${stepOf('ensemble')}/${totalModels}: Voting Ensemble...`);
            await new Promise(r => setTimeout(r, 60));
            const ensembleConfig = {
                voting: 'soft',
                // Same base models for classification and regression (consistent with the Step 4 default)
                baseModels: ['nn', 'rf', 'glm'],
                capacity: 'balanced',
                seed: AppState.pipeline.evalSeed
            };
            const ensemble = new EnsembleModel(isClassification, numClasses, ensembleConfig);
            await ensemble.fit(fullPrep.X, fullPrep.y, null, null, (curr, tot, info) => {
                if (watchdog) {
                    watchdog.tick(stepOf('ensemble'), `Training Model ${stepOf('ensemble')}/${totalModels}: Ensemble (${info.stage || `Base ${curr}/${tot}`})...`);
                }
            });
            const ensEval = ModelEngine.evaluate(ensemble, fullTestTrans.X, fullTestTrans.y, isClassification, fullPrep.targetClasses, pipeEvalOptions);
            ensEval.featuresUsed = `All Surviving (${activeCols.length} features)`;
            if (ensemble.fallbackNote) ensEval.fallbackNote = ensemble.fallbackNote;
            ensEval.constraintNote = `Equal-weight soft vote of ${ensembleConfig.baseModels.map(b => b.toUpperCase()).join(', ')}; scaling fitted on Train only.`;
            benchmarkList.push({ id: 'ensemble', name: 'Voting / Weighted Ensemble', model: ensemble, prep: fullPrep, eval: ensEval });

            // Store ranked best-first (macro-F1 then AUC for classification; R² for regression),
            // so benchmarkResults[0] is genuinely the top model everywhere it is used
            benchmarkList.sort((a, b) => compareEvalResults(a.eval, b.eval, isClassification));
            disposeModels((AppState.pipeline.benchmarkResults || []).map(b => b.model));
            AppState.pipeline.benchmarkResults = benchmarkList;

            // Render Leaderboard & Confusion Matrices
            renderPipelineBenchmarkResults(benchmarkList, fullPrep.targetClasses, isClassification);

            if (DOM.pipelineResultsContainer) DOM.pipelineResultsContainer.style.display = 'block';

            if (watchdog) watchdog.finish(`All ${benchmarkList.length} Clinical Models Evaluated Successfully!`);

        } catch (err) {
            console.error("Pipeline Step 7 training error:", err);
            if (watchdog && !watchdog.aborted) {
                watchdog.forceAbort(`Pipeline error: ${err.message}`);
            }
            alert(`Pipeline training error: ${err.message}`);
        } finally {
            if (DOM.pipelineTrainModelsBtn) DOM.pipelineTrainModelsBtn.disabled = false;
        }
    }

    function renderPipelineLeaderboardOnly() {
        if (!AppState.pipeline.benchmarkResults || !DOM.pipelineLeaderboardContainer) return;
        const isClassification = AppState.pipeline.columnConfigs[AppState.pipeline.targetCol]?.type !== 'continuous';
        const sorted = [...AppState.pipeline.benchmarkResults].sort((a, b) => compareEvalResults(a.eval, b.eval, isClassification));

        let tableHtml = `
            <p style="font-size: 12px; color: var(--text-secondary); margin-bottom: 8px;">
                Ranked by <strong>${escapeHtml(rankingMetricLabel(isClassification))}</strong> on the held-out ${Math.round((1 - (AppState.pipeline.trainRatio || 0.8)) * 100)}% test split (change the ranking metric above).
                The top model is selected on these same test cases, so its score is optimistically biased; confirm on an independent cohort.
            </p>`;
        tableHtml += `
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Algorithm</th>
                        <th>Features Subset</th>
                        <th>${isClassification ? 'F1-Score (Macro)' : 'R²'}</th>
                        <th>${isClassification ? 'Precision' : 'MAE'}</th>
                        <th>${isClassification ? 'Recall' : 'RMSE'}</th>
                        <th>AUC-ROC</th>
                        <th>${isClassification ? 'Balanced Acc' : '-'}</th>
                        <th>${isClassification ? 'Brier ↓' : '-'}</th>
                        <th>${isClassification ? 'Test Accuracy' : 'MSE'}</th>
                        <th>Algorithm Constraint</th>
                        <th style="text-align: right; width: 140px;">Deployment</th>
                    </tr>
                </thead>
                <tbody>
        `;

        sorted.forEach(item => {
            const ev = item.eval;
            const f1 = isClassification ? (ev.macroF1 !== undefined ? (ev.macroF1 * 100).toFixed(1) + '%' : 'N/A') : fmtR2(ev.r2);
            const prec = isClassification ? (ev.macroPrecision !== undefined ? (ev.macroPrecision * 100).toFixed(1) + '%' : 'N/A') : `${ev.mae}`;
            const rec = isClassification ? (ev.macroRecall !== undefined ? (ev.macroRecall * 100).toFixed(1) + '%' : 'N/A') : `${ev.rmse}`;
            const auc = isClassification ? (ev.rocAuc !== null && ev.rocAuc !== undefined ? ev.rocAuc.toFixed(3) : 'N/A') : 'N/A';
            const acc = isClassification ? (ev.accuracy !== undefined ? (ev.accuracy * 100).toFixed(1) + '%' : 'N/A') : `${ev.mse}`;
            const bal = isClassification && typeof ev.balancedAccuracy === 'number' ? (ev.balancedAccuracy * 100).toFixed(1) + '%' : '-';
            const brier = isClassification && typeof ev.brierScore === 'number' ? ev.brierScore.toFixed(3) : '-';

            const isDeployed = AppState.deployedSource === '7step' && AppState.deployedModelId === item.id;

            tableHtml += `
                <tr style="${isDeployed ? 'background: rgba(16, 185, 129, 0.05);' : ''}">
                    <td><strong>${escapeHtml(item.name)}</strong></td>
                    <td><span class="pill pill-target" style="font-size: 11px;">${ev.featuresUsed}</span></td>
                    <td><strong style="color: var(--accent-emerald); font-size: 13.5px;">${f1}</strong></td>
                    <td>${prec}</td>
                    <td>${rec}</td>
                    <td><strong style="color: var(--accent-cyan);">${auc}</strong></td>
                    <td>${bal}</td>
                    <td>${brier}</td>
                    <td><strong>${acc}</strong></td>
                    <td style="font-size: 11.5px; color: var(--text-secondary);">${escapeHtml(ev.constraintNote)}</td>
                    <td style="text-align: right;">
                        ${isDeployed ? `
                            <span class="pill" style="background: rgba(16, 185, 129, 0.15); color: var(--accent-emerald); font-weight: 700; border-color: rgba(16, 185, 129, 0.3);">
                                ✓ Active
                            </span>
                        ` : `
                            <button class="btn btn-secondary btn-sm" onclick="window.deployPipelineModel('${item.id}')" title="Deploy this model for live patient inference">
                                🚀 Deploy
                            </button>
                        `}
                    </td>
                </tr>
            `;
        });

        tableHtml += `</tbody></table>`;
        const caveats = new Set();
        sorted.forEach(item => {
            summariseHeldOutSet(item.eval).forEach(n => caveats.add(n));
            if (item.eval.fallbackNote) caveats.add(`⚠️ ${escapeHtml(item.eval.fallbackNote)}`);
        });
        if (isClassification && sorted[0] && sorted[0].eval.positiveClass !== null && sorted[0].eval.positiveClass !== undefined) {
            caveats.add(`Positive (event) class for AUC, Brier, sensitivity and PPV: <strong>${escapeHtml(sorted[0].eval.positiveClass)}</strong> (set in Step 2).`);
        }
        if (caveats.size > 0) {
            tableHtml += `<div style="font-size: 11.5px; color: var(--text-secondary); margin-top: 10px; padding: 8px 12px; border-left: 3px solid var(--accent-amber); background: var(--bg-surface-subtle);">
                ${[...caveats].map(c => `<div>• ${c}</div>`).join('')}
            </div>`;
        }
        DOM.pipelineLeaderboardContainer.innerHTML = tableHtml;
        syncRankingSelectors(isClassification);
    }

    function renderPipelineBenchmarkResults(benchmarkList, targetClasses, isClassification) {
        // Sort best-first (same criterion as the leaderboard)
        const sorted = [...benchmarkList].sort((a, b) => compareEvalResults(a.eval, b.eval, isClassification));

        // 1. Leaderboard Table
        renderPipelineLeaderboardOnly();

        // 2. 2x2 Confusion Matrices Grid
        let cmHtml = '';
        sorted.forEach(item => {
            const ev = item.eval;
            const cm = ev.confMatrix;

            if (!isClassification) {
                // Regression: report error metrics (a median-split 2×2 table is shown in Step 4 only as a descriptive aid)
                cmHtml += `
                    <div class="card" style="padding: 16px; background: var(--bg-surface-subtle); border: 1px solid var(--border-color);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <strong style="font-size: 13.5px; color: var(--text-primary);">${escapeHtml(item.name)}</strong>
                            <span class="pill pill-continuous" style="font-size: 10.5px;">R²: ${fmtR2(ev.r2)}</span>
                        </div>
                        <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 10px;">
                            ${escapeHtml(ev.featuresUsed)} &bull; ${escapeHtml(ev.constraintNote)}
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; font-size: 11px; text-align: center;">
                            <div style="background: var(--bg-surface); padding: 4px; border-radius: 4px;">RMSE: <strong>${ev.rmse}</strong></div>
                            <div style="background: var(--bg-surface); padding: 4px; border-radius: 4px;">MAE: <strong>${ev.mae}</strong></div>
                            <div style="background: var(--bg-surface); padding: 4px; border-radius: 4px;">MSE: <strong>${ev.mse}</strong></div>
                        </div>
                    </div>
                `;
            } else if (cm && cm.length === 2) {
                // Positive class from the evaluation result (set in Step 2); index lookup keeps TP/FN/FP/TN
                // correct even if the class order ever differs from [negative, positive]
                const posLabel = ev.positiveClass !== null && ev.positiveClass !== undefined ? ev.positiveClass : targetClasses[1];
                const posIdx = targetClasses.indexOf(posLabel) === 0 ? 0 : 1;
                const negIdx = 1 - posIdx;
                const negLabel = targetClasses[negIdx];
                const tn = cm[negIdx][negIdx];
                const fp = cm[negIdx][posIdx];
                const fn = cm[posIdx][negIdx];
                const tp = cm[posIdx][posIdx];
                // Undefined ratios (zero denominator) are shown as N/A rather than 0%
                const sens = (tp + fn) > 0 ? ((tp / (tp + fn)) * 100).toFixed(1) + '%' : 'N/A';
                const spec = (tn + fp) > 0 ? ((tn / (tn + fp)) * 100).toFixed(1) + '%' : 'N/A';
                const ppv = (tp + fp) > 0 ? ((tp / (tp + fp)) * 100).toFixed(1) + '%' : 'N/A';
                const npv = (tn + fn) > 0 ? ((tn / (tn + fn)) * 100).toFixed(1) + '%' : 'N/A';

                cmHtml += `
                    <div class="card" style="padding: 16px; background: var(--bg-surface-subtle); border: 1px solid var(--border-color);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <strong style="font-size: 13.5px; color: var(--text-primary);">${escapeHtml(item.name)}</strong>
                            <span class="pill pill-continuous" style="font-size: 10.5px;">Acc: ${(ev.accuracy * 100).toFixed(1)}%</span>
                        </div>
                        <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 10px;">
                            ${escapeHtml(ev.featuresUsed)} &bull; ${escapeHtml(ev.constraintNote)}
                        </div>
                        <div style="font-size: 10.5px; color: var(--text-muted); margin-bottom: 6px;">
                            "Positive" = <strong>${escapeHtml(posLabel)}</strong> (the positive/event class chosen in Step 2); check this matches your clinical event of interest.
                        </div>
                        <div class="cm-table-wrapper" style="margin-bottom: 10px;">
                            <table class="cm-table">
                                <thead>
                                    <tr>
                                        <th>Actual \\ Pred</th>
                                        <th>Negative (${escapeHtml(negLabel)})</th>
                                        <th>Positive (${escapeHtml(posLabel)})</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td><strong>Negative (${escapeHtml(negLabel)})</strong></td>
                                        <td class="cm-cell-tn">TN: <strong>${tn}</strong></td>
                                        <td class="cm-cell-fp">FP: <strong>${fp}</strong></td>
                                    </tr>
                                    <tr>
                                        <td><strong>Positive (${escapeHtml(posLabel)})</strong></td>
                                        <td class="cm-cell-fn">FN: <strong>${fn}</strong></td>
                                        <td class="cm-cell-tp">TP: <strong>${tp}</strong></td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; font-size: 11px; text-align: center;">
                            <div style="background: var(--bg-surface); padding: 4px; border-radius: 4px;">Sens: <strong>${sens}</strong></div>
                            <div style="background: var(--bg-surface); padding: 4px; border-radius: 4px;">Spec: <strong>${spec}</strong></div>
                            <div style="background: var(--bg-surface); padding: 4px; border-radius: 4px;">PPV: <strong>${ppv}</strong></div>
                            <div style="background: var(--bg-surface); padding: 4px; border-radius: 4px;">NPV: <strong>${npv}</strong></div>
                        </div>
                    </div>
                `;
            } else if (cm && cm.length > 2) {
                // Multiclass outcome: full K×K matrix (rows = actual, columns = predicted)
                cmHtml += `
                    <div class="card" style="padding: 16px; background: var(--bg-surface-subtle); border: 1px solid var(--border-color);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <strong style="font-size: 13.5px; color: var(--text-primary);">${escapeHtml(item.name)}</strong>
                            <span class="pill pill-continuous" style="font-size: 10.5px;">Acc: ${(ev.accuracy * 100).toFixed(1)}%</span>
                        </div>
                        <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 10px;">
                            ${escapeHtml(ev.featuresUsed)} &bull; ${escapeHtml(ev.constraintNote)}
                        </div>
                        <div class="cm-table-wrapper" style="margin-bottom: 10px; overflow-x: auto;">
                            <table class="cm-table" style="font-size: 11.5px;">
                                <thead>
                                    <tr>
                                        <th>Actual \\ Pred</th>
                                        ${targetClasses.map(c => `<th>${escapeHtml(c)}</th>`).join('')}
                                    </tr>
                                </thead>
                                <tbody>
                                    ${targetClasses.map((actual, r) => `
                                        <tr>
                                            <th style="text-align: left;">${escapeHtml(actual)}</th>
                                            ${targetClasses.map((_, c) => `<td class="${r === c ? 'cm-cell-hit' : 'cm-cell-miss'}">${cm[r] && cm[r][c] !== undefined ? cm[r][c] : 0}</td>`).join('')}
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                        <div style="font-size: 11px; color: var(--text-secondary);">
                            Macro F1: <strong>${(ev.macroF1 * 100).toFixed(1)}%</strong> &bull;
                            Balanced accuracy: <strong>${typeof ev.balancedAccuracy === 'number' ? (ev.balancedAccuracy * 100).toFixed(1) + '%' : 'N/A'}</strong>
                            (computed over all ${targetClasses.length} classes; sensitivity/PPV apply per class, not to the whole table)
                        </div>
                    </div>
                `;
            }
        });

        if (DOM.pipelineCmGrid) DOM.pipelineCmGrid.innerHTML = cmHtml;

        // Auto-deploy top-performing model if nothing deployed yet or previous deployment was from pipeline
        if (!AppState.trainedModel || AppState.deployedSource === '7step') {
            deployPipelineModel(sorted[0].id, false, true); // skipRerender = true to prevent recursion
        } else {
            updatePipelineActiveDeploymentStatus();
            updatePredictionProvenanceBar();
        }
    }

    function deployPipelineModel(modelId, showNotice = true, skipRerender = false) {
        if (!AppState.pipeline.benchmarkResults) return;
        const targetItem = AppState.pipeline.benchmarkResults.find(b => b.id === modelId);
        if (!targetItem) return;

        AppState.trainedModel = targetItem.model;
        AppState.prepMeta = targetItem.prep;
        AppState.evalResults = targetItem.eval;
        AppState.deployedModelId = targetItem.id;
        AppState.deployedSource = '7step';
        AppState.deployedPipelineModel = targetItem;

        // Enable Step 5 & Step 6 buttons
        if (DOM.toStep5Btn) DOM.toStep5Btn.disabled = false;
        if (DOM.toStep6Btn) DOM.toStep6Btn.disabled = false;

        updatePipelineActiveDeploymentStatus();
        updatePredictionProvenanceBar();
        renderSinglePredictionForm();

        // Update leaderboard row styling without recursive mutual autoDeploy
        if (!skipRerender) {
            renderPipelineLeaderboardOnly();
        }

        if (showNotice) {
            showSeedToast(`✓ Deployed "${targetItem.name}" as the active predictor for Steps 5 & 6`);
        }
    }
    window.deployPipelineModel = deployPipelineModel;

    function autoDeployBestPipelineModel() {
        if (!AppState.pipeline.benchmarkResults || AppState.pipeline.benchmarkResults.length === 0) return;
        const isClassification = AppState.pipeline.columnConfigs[AppState.pipeline.targetCol]?.type !== 'continuous';
        const sorted = [...AppState.pipeline.benchmarkResults].sort((a, b) => compareEvalResults(a.eval, b.eval, isClassification));
        deployPipelineModel(sorted[0].id, false);
    }

    function updatePipelineActiveDeploymentStatus() {
        if (!DOM.pipelineActiveDeploymentStatus) return;
        if (AppState.deployedPipelineModel) {
            DOM.pipelineActiveDeploymentStatus.innerHTML = `
                Currently Active Deployed Model: <strong style="color: var(--accent-emerald);">${escapeHtml(AppState.deployedPipelineModel.name)}</strong> 
                (${escapeHtml(AppState.deployedPipelineModel.eval.featuresUsed)}). 
                ${pipelineProvenanceSummary()} &bull; Ready for single-case and batch predictions (research use).
            `;
        } else {
            DOM.pipelineActiveDeploymentStatus.innerHTML = `Select any model from the leaderboard to deploy for real-time patient predictions.`;
        }
    }

    // Factual one-line summary of what the 7-step pipeline actually did
    function pipelineProvenanceSummary() {
        const p = AppState.pipeline;
        const parts = ['Imputation & scaling fitted on training split only'];
        const vifDropped = (p.vifResults || []).filter(v => v.highVIF && p.droppedFeatures.has(v.feature)).length;
        parts.push(p.droppedFeatures.size > 0 ? `${p.droppedFeatures.size} feature(s) excluded (correlation/VIF/univariate review)` : 'no features excluded');
        if (vifDropped > 0) parts.push(`${vifDropped} with VIF > 5`);
        if (p.smotencResult) {
            parts.push(p.smotencResult.applied ? `SMOTENC applied (+${p.smotencResult.syntheticCountGenerated} synthetic training rows)` : 'SMOTENC not applied');
        }
        return parts.join(' &bull; ');
    }

    function updatePredictionProvenanceBar() {
        const singleBar = DOM.singlePredictProvenanceBar;
        const batchBar = DOM.batchPredictProvenanceBar;
        if (!singleBar && !batchBar) return;

        const hasPipeline = AppState.pipeline && AppState.pipeline.benchmarkResults && AppState.pipeline.benchmarkResults.length > 0;
        const hasBaseline = AppState.allTestedResults && AppState.allTestedResults.length > 0;

        let contentHtml = '';
        if (AppState.deployedSource === '7step' && AppState.deployedPipelineModel) {
            const m = AppState.deployedPipelineModel;
            const featCount = AppState.prepMeta ? AppState.prepMeta.featureCols.length : (AppState.pipeline.top70Features?.length || 'Optimal');
            contentHtml = `
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 22px;">🔬</span>
                    <div>
                        <div style="font-size: 13.5px; font-weight: 700; color: var(--accent-emerald);">
                            Active Predictor: ${escapeHtml(m.name)} &bull; Trained via 7-Step Clinical Pipeline
                        </div>
                        <div style="font-size: 11.5px; color: var(--text-secondary); margin-top: 2px;">
                            ${pipelineProvenanceSummary()} &bull; ${featCount} active features
                        </div>
                    </div>
                </div>
                <div style="display: flex; gap: 8px; align-items: center;">
                    ${hasBaseline ? `<button class="btn btn-secondary btn-sm" onclick="window.switchToBaselineModel()">🔄 Switch to Baseline Model</button>` : ''}
                    <button class="btn btn-secondary btn-sm" onclick="window.goToStep && window.goToStep('pipeline')">🔬 View 7-Step Pipeline</button>
                </div>
            `;
            if (singleBar) {
                singleBar.className = 'pipeline-alert pipeline-alert-success';
                singleBar.style.display = 'flex';
                singleBar.innerHTML = contentHtml;
            }
            if (batchBar) {
                batchBar.className = 'pipeline-alert pipeline-alert-success';
                batchBar.style.display = 'flex';
                batchBar.innerHTML = contentHtml;
            }
        } else if (AppState.trainedModel) {
            const mInfo = MODEL_INFO[AppState.deployedModelId] || { name: (AppState.deployedModelId || 'ML').toUpperCase() };
            const featCount = AppState.prepMeta ? AppState.prepMeta.featureCols.length : 'All';
            contentHtml = `
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 22px;">📋</span>
                    <div>
                        <div style="font-size: 13.5px; font-weight: 700; color: var(--accent-cyan);">
                            Active Predictor: ${mInfo.name} &bull; Trained via Pre-existing Standard Workflow
                        </div>
                        <div style="font-size: 11.5px; color: var(--text-secondary); margin-top: 2px;">
                            Standard Workflow &bull; ${AppState.evalResults && AppState.evalResults.protocol === 'option-b' ? 'Preprocessing fitted on Dataset (i)' : (AppState.evalResults && AppState.evalResults.retrainOn100 ? 'Final model refitted on the full cohort after split-first evaluation' : 'Preprocessing fitted on the training split only')} &bull; ${featCount} features
                        </div>
                    </div>
                </div>
                <div style="display: flex; gap: 8px; align-items: center;">
                    ${hasPipeline ? `<button class="btn btn-emerald btn-sm" onclick="window.switchToPipelineModel()">🔬 Switch to 7-Step Pipeline Model</button>` : `<button class="btn btn-emerald btn-sm" onclick="window.goToStep && window.goToStep('pipeline')">🔬 Test 7-Step Pipeline to Compare</button>`}
                </div>
            `;
            if (singleBar) {
                singleBar.className = 'pipeline-alert pipeline-alert-info';
                singleBar.style.display = 'flex';
                singleBar.innerHTML = contentHtml;
            }
            if (batchBar) {
                batchBar.className = 'pipeline-alert pipeline-alert-info';
                batchBar.style.display = 'flex';
                batchBar.innerHTML = contentHtml;
            }
        } else {
            if (singleBar) singleBar.style.display = 'none';
            if (batchBar) batchBar.style.display = 'none';
        }
    }

    window.switchToBaselineModel = function () {
        if (!AppState.allTestedResults || AppState.allTestedResults.length === 0) {
            alert("No baseline model found. Train a model in Step 4 first.");
            return;
        }
        const bestBaseline = AppState.allTestedResults[0];
        deployTestedModel(bestBaseline.modelId);
        updatePredictionProvenanceBar();
    };

    window.switchToPipelineModel = function () {
        if (!AppState.pipeline.benchmarkResults || AppState.pipeline.benchmarkResults.length === 0) {
            alert("7-Step Pipeline has not been run yet. Please run the 7-Step Pipeline first.");
            if (window.goToStep) window.goToStep('pipeline');
            return;
        }
        const bestPipeline = AppState.pipeline.benchmarkResults[0];
        deployPipelineModel(bestPipeline.id, true);
    };

    function togglePipelineBaselineComparison() {
        if (!DOM.pipelineVsBaselinePanel) return;
        const isHidden = DOM.pipelineVsBaselinePanel.style.display === 'none';
        if (isHidden) {
            DOM.pipelineVsBaselinePanel.style.display = 'block';
            renderPipelineVsBaselineComparison();
            DOM.pipelineVsBaselinePanel.scrollIntoView({ behavior: 'smooth' });
        } else {
            DOM.pipelineVsBaselinePanel.style.display = 'none';
        }
    }

    window.trainStandardBaselineFromPipeline = function () {
        if (!AppState.parsedData || !AppState.parsedData.rows || AppState.parsedData.rows.length === 0) {
            alert("No dataset loaded yet. Please select a sample cohort or upload a CSV first.");
            return;
        }

        // If Step 3 imputation has not been run yet, run standard imputation automatically
        if (!AppState.imputedData) {
            try {
                executeImputation();
            } catch (err) {
                console.warn("Auto-imputation error:", err);
            }
        }

        // Navigate directly to Step 4
        goToStep(4);
        window.scrollTo({ top: 0, behavior: 'smooth' });

        // Scroll to train button with visual highlight
        setTimeout(() => {
            const trainBtn = document.getElementById('train-model-btn');
            if (trainBtn) {
                trainBtn.scrollIntoView({ behavior: 'smooth', block: 'center' });
                trainBtn.style.boxShadow = '0 0 0 3px var(--accent-emerald)';
                setTimeout(() => { if (trainBtn) trainBtn.style.boxShadow = ''; }, 2500);
            }
        }, 150);
    };

    function renderPipelineVsBaselineComparison() {
        if (!DOM.pipelineVsBaselineContent) return;

        const hasPipeline = AppState.pipeline && AppState.pipeline.benchmarkResults && AppState.pipeline.benchmarkResults.length > 0;
        const hasBaseline = AppState.allTestedResults && AppState.allTestedResults.length > 0;

        if (!hasPipeline) {
            DOM.pipelineVsBaselineContent.innerHTML = `
                <p style="padding: 16px; color: var(--text-muted);">Please complete Step 7 of the pipeline first to generate metrics.</p>
            `;
            return;
        }

        const pipeBest = AppState.pipeline.benchmarkResults[0]; // stored best-first in executePipelineStep7
        const pEv = pipeBest.eval;

        if (!hasBaseline) {
            const f1Text = (pEv && typeof pEv.macroF1 === 'number' && !isNaN(pEv.macroF1))
                ? `F1: ${(pEv.macroF1 * 100).toFixed(1)}%`
                : (pEv && typeof pEv.r2 === 'number' && !isNaN(pEv.r2) ? `R²: ${pEv.r2.toFixed(3)}` : '');
            const accText = (pEv && typeof pEv.accuracy === 'number' && !isNaN(pEv.accuracy))
                ? `Acc: ${(pEv.accuracy * 100).toFixed(1)}%`
                : (pEv && typeof pEv.rmse === 'number' && !isNaN(pEv.rmse) ? `RMSE: ${pEv.rmse.toFixed(3)}` : '');
            const metricSummary = [f1Text, accText].filter(Boolean).join(', ') || 'Evaluated';

            DOM.pipelineVsBaselineContent.innerHTML = `
                <div class="pipeline-alert pipeline-alert-info">
                    <div style="font-size: 20px;">💡</div>
                    <div>
                        <strong>Baseline Benchmark Not Run Yet:</strong><br>
                        You have successfully completed the 7-Step Pipeline with <strong>${escapeHtml(pipeBest.name)}</strong> (${metricSummary}).<br>
                        To see if the 7-Step Pipeline improved predictions compared to the standard pre-existing workflow on this exact cohort, run Step 4 (Model Training) or click below to launch the standard baseline:
                        <div style="margin-top: 10px;">
                            <button class="btn btn-primary btn-sm" onclick="window.trainStandardBaselineFromPipeline && window.trainStandardBaselineFromPipeline()">
                                🏃 Train Standard Baseline in Step 4
                            </button>
                        </div>
                    </div>
                </div>
            `;
            return;
        }

        // Both baseline and pipeline exist: side-by-side (unpaired) comparison
        const baseBest = AppState.allTestedResults[0];
        const bEv = baseBest.evalRes;
        const isClassification = AppState.pipeline.columnConfigs[AppState.pipeline.targetCol]?.type !== 'continuous';

        if (Boolean(bEv.isClassification) !== Boolean(isClassification)) {
            DOM.pipelineVsBaselineContent.innerHTML = `
                <div class="pipeline-alert pipeline-alert-info">
                    <div style="font-size: 20px;">ℹ️</div>
                    <div>The Step 4 baseline and the 7-Step Pipeline were run with different outcome types (classification vs regression), so they cannot be compared. Re-run one of them with the same target configuration.</div>
                </div>`;
            return;
        }

        const fmtDelta = (d, digits, unit) => {
            if (d === null || isNaN(d)) return '—';
            const sign = d > 0 ? '+' : '';
            return `<strong style="color: ${d > 0 ? 'var(--accent-emerald)' : (d < 0 ? 'var(--accent-rose)' : 'var(--text-secondary)')};">${sign}${d.toFixed(digits)}${unit}</strong>`;
        };
        const row = (label, bVal, pVal, delta) => `
            <tr>
                <td><strong>${label}</strong></td>
                <td>${bVal}</td>
                <td><strong>${pVal}</strong></td>
                <td>${delta}</td>
            </tr>`;

        let metricRows = '';
        let headlineDelta = null;
        let headlineLabel = '';
        if (isClassification) {
            const pct = v => (typeof v === 'number' && !isNaN(v) ? v * 100 : null);
            const bF1 = pct(bEv.macroF1), pF1 = pct(pEv.macroF1);
            const bAcc = pct(bEv.accuracy), pAcc = pct(pEv.accuracy);
            const bAuc = typeof bEv.rocAuc === 'number' ? bEv.rocAuc : null;
            const pAuc = typeof pEv.rocAuc === 'number' ? pEv.rocAuc : null;
            const dF1 = (bF1 !== null && pF1 !== null) ? pF1 - bF1 : null;
            headlineDelta = dF1;
            headlineLabel = 'macro F1';
            metricRows += row('Macro F1-Score', bF1 !== null ? `${bF1.toFixed(1)}%` : 'N/A', pF1 !== null ? `${pF1.toFixed(1)}%` : 'N/A', fmtDelta(dF1, 1, ' pp'));
            metricRows += row('Test Accuracy', bAcc !== null ? `${bAcc.toFixed(1)}%` : 'N/A', pAcc !== null ? `${pAcc.toFixed(1)}%` : 'N/A',
                fmtDelta((bAcc !== null && pAcc !== null) ? pAcc - bAcc : null, 1, ' pp'));
            if (bAuc !== null && pAuc !== null) {
                metricRows += row('ROC-AUC (binary)', bAuc.toFixed(3), pAuc.toFixed(3), fmtDelta(pAuc - bAuc, 3, ''));
            }
        } else {
            const num = v => (typeof v === 'number' && !isNaN(v) ? v : null);
            const bR2 = num(bEv.r2), pR2 = num(pEv.r2);
            const bRmse = num(bEv.rmse), pRmse = num(pEv.rmse);
            const bMae = num(bEv.mae), pMae = num(pEv.mae);
            headlineDelta = (bR2 !== null && pR2 !== null) ? pR2 - bR2 : null;
            headlineLabel = 'R²';
            metricRows += row('R²', bR2 !== null ? bR2.toFixed(3) : 'N/A', pR2 !== null ? pR2.toFixed(3) : 'N/A', fmtDelta(headlineDelta, 3, ''));
            // Lower error is better: show (baseline − pipeline) so positive means the pipeline is better
            metricRows += row('RMSE (lower is better)', bRmse !== null ? bRmse.toFixed(3) : 'N/A', pRmse !== null ? pRmse.toFixed(3) : 'N/A',
                fmtDelta((bRmse !== null && pRmse !== null) ? bRmse - pRmse : null, 3, ' reduction'));
            metricRows += row('MAE (lower is better)', bMae !== null ? bMae.toFixed(3) : 'N/A', pMae !== null ? pMae.toFixed(3) : 'N/A',
                fmtDelta((bMae !== null && pMae !== null) ? bMae - pMae : null, 3, ' reduction'));
        }

        const sm = AppState.pipeline.smotencResult;
        const baseProtocol = bEv.protocol === 'option-b'
            ? `external Dataset (ii) (${bEv.testCount} cases)`
            : `${bEv.testPct || 20}% random split (${bEv.testCount} cases, split seed ${bEv.splitSeed})`;
        const pipeTestCount = AppState.pipeline.testRows ? AppState.pipeline.testRows.length : '?';

        let headline;
        if (headlineDelta === null) {
            headline = 'The headline metric could not be compared.';
        } else if (Math.abs(headlineDelta) < 1e-9) {
            headline = `Both workflows obtained the same ${headlineLabel}.`;
        } else {
            const unit = isClassification ? ' percentage points' : '';
            headline = `The 7-Step Pipeline's top model scored ${Math.abs(headlineDelta).toFixed(isClassification ? 1 : 3)}${unit} ${headlineDelta > 0 ? 'higher' : 'lower'} ${headlineLabel} than the Step 4 baseline.`;
        }

        const compHtml = `
            <div style="margin-bottom: 14px; font-size: 13px; color: var(--text-secondary);">
                Side-by-side results for cohort <strong>${escapeHtml(AppState.fileName || 'Cohort')}</strong>.
                <span style="color: var(--accent-amber);">This is not a paired comparison:</span> the baseline was scored on a ${baseProtocol},
                the pipeline on its own ${Math.round((AppState.pipeline.trainRatio || 0.8) * 100)}/${Math.round((1 - (AppState.pipeline.trainRatio || 0.8)) * 100)} split (${pipeTestCount} test cases, seed ${AppState.pipeline.seed}). Both "top" models were chosen on their own test sets.
                Differences of a few points can arise from split variation alone; no significance test has been applied.
            </div>

            <table class="data-table" style="margin-bottom: 16px;">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th style="color: var(--accent-cyan);">Step 4 Standard Workflow</th>
                        <th style="color: var(--accent-emerald);">🔬 7-Step Pipeline</th>
                        <th>Difference (Pipeline − Baseline)</th>
                    </tr>
                </thead>
                <tbody>
                    ${row('Top-ranked model', escapeHtml(MODEL_INFO[baseBest.modelId]?.name || baseBest.modelId.toUpperCase()), escapeHtml(pipeBest.name), '')}
                    ${metricRows}
                    ${row('Imputation / scaling', bEv.protocol === 'option-b' ? 'Fitted on Dataset (i) only' : 'Fitted on the training split only (split first)', 'Fitted on the training split only', '')}
                    ${row('Collinearity handling', '<span style="color: var(--text-muted);">None (all features kept)</span>', `Correlation & VIF review; ${AppState.pipeline.droppedFeatures.size} feature(s) excluded overall`, '')}
                    ${row('Class imbalance', '<span style="color: var(--text-muted);">Original distribution</span>',
                        isClassification ? (sm && sm.applied ? `SMOTENC on training split (+${sm.syntheticCountGenerated} synthetic rows)` : 'SMOTENC not applied (not imbalanced)') : 'Not applicable (regression)', '')}
                    ${row('Feature pruning', '<span style="color: var(--text-muted);">All features</span>', 'Bottom 30% RF-importance features removed for k-NN & Naive Bayes only', '')}
                </tbody>
            </table>

            <div class="pipeline-alert pipeline-alert-info" style="margin-top: 10px;">
                <div style="font-size: 20px;">🔬</div>
                <div>
                    <strong>Summary:</strong> ${headline} To establish a genuine difference, compare both workflows on the same independent validation cohort or with repeated cross-validation.
                </div>
            </div>
        `;

        DOM.pipelineVsBaselineContent.innerHTML = compHtml;
    }

    function downloadPipelineProtocolReport() {
        if (!AppState.pipeline.benchmarkResults || AppState.pipeline.benchmarkResults.length === 0) {
            alert("No pipeline results available to download.");
            return;
        }

        const report = {
            title: "Strict 7-Step Clinical Machine Learning Protocol Report",
            conceived_and_directed_by: "Dr G Narenthiran MB ChB BSc(MedSci)(Hons) FEBNS FRCS(SN)",
            generated_at: new Date().toISOString(),
            dataset_name: AppState.fileName,
            target_variable: AppState.pipeline.targetCol,
            reproducibility_seed: AppState.pipeline.evalSeed,
            pipeline_steps: {
                step_1_split_and_strict_imputation: {
                    train_records: AppState.pipeline.trainRows.length,
                    test_records: AppState.pipeline.testRows.length,
                    rows_excluded_missing_outcome: AppState.pipeline.excludedMissingTarget || 0,
                    split_seed: AppState.pipeline.seed,
                    imputation_strategy: AppState.pipeline.imputeStrategy,
                    leak_free_audit: AppState.pipeline.imputer.fillValues
                },
                step_2_skewness_and_correlation: {
                    continuous_features_analyzed: Object.keys(AppState.pipeline.normalityMap),
                    skewness_screen_rule: '|g1| <= 1 treated as approximately symmetric (rule of thumb, not a formal normality test)',
                    skewness_assessment: AppState.pipeline.normalityMap,
                    high_correlation_pairs: AppState.pipeline.corrPairs.filter(p => p.isHigh)
                },
                step_3_vif_check: {
                    vif_scores: AppState.pipeline.vifResults,
                    excluded_features_all_steps: Array.from(AppState.pipeline.droppedFeatures)
                },
                step_4_univariate_testing: {
                    baseline_rankings: AppState.pipeline.univariateResults
                },
                step_5_smotenc_imbalance_handling: {
                    applied: AppState.pipeline.smotencResult?.applied,
                    reason: AppState.pipeline.smotencResult?.reason,
                    minority_percentage_before: AppState.pipeline.smotencResult?.minorityPctBefore,
                    minority_percentage_after: AppState.pipeline.smotencResult?.minorityPctAfter,
                    synthetic_instances_generated: AppState.pipeline.smotencResult?.syntheticCountGenerated || 0
                },
                step_6_rf_feature_scout: {
                    feature_importance_scores: AppState.pipeline.rfImportance,
                    top_70_features: AppState.pipeline.top70Features,
                    bottom_30_features: AppState.pipeline.bottom30Features
                },
                step_7_ranking_metric: rankingMetricLabel(AppState.pipeline.columnConfigs[AppState.pipeline.targetCol]?.type !== 'continuous'),
                step_7_model_selection_note: 'Models are ranked best-first on the held-out test split; the top model is selected on that same split, so its score is optimistically biased. A single split does not establish general superiority.',
                step_7_constrained_model_benchmark: AppState.pipeline.benchmarkResults.map(b => ({
                    model_id: b.id,
                    model_name: b.name,
                    features_used: b.eval.featuresUsed,
                    constraint_note: b.eval.constraintNote,
                    neural_network_fell_back_to_linear_model: Boolean(b.model && b.model.usedFallback),
                    nn_training: b.model && b.model.trainingInfo ? b.model.trainingInfo : undefined,
                    metrics: b.eval.isClassification ? {
                        f1_score_macro: b.eval.macroF1,
                        precision_macro: b.eval.macroPrecision,
                        recall_macro: b.eval.macroRecall,
                        balanced_accuracy: b.eval.balancedAccuracy,
                        auc_roc: b.eval.rocAuc,
                        brier_score: b.eval.brierScore,
                        accuracy: b.eval.accuracy,
                        accuracy_95ci_wilson: b.eval.accuracyCI,
                        probability_type: b.eval.probabilityType,
                        positive_class: b.eval.positiveClass,
                        classes_missing_from_test: b.eval.classesMissingFromTest
                    } : {
                        r2: b.eval.r2,
                        rmse: b.eval.rmse,
                        mae: b.eval.mae,
                        mse: b.eval.mse,
                        exploratory_cutoff: b.eval.threshold,
                        exploratory_cutoff_source: b.eval.thresholdSource
                    },
                    confusion_matrix: b.eval.confMatrix
                }))
            }
        };

        const jsonString = JSON.stringify(report, null, 2);
        const blob = new Blob([jsonString], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `Clinical_ML_7Step_Protocol_Report_${new Date().toISOString().slice(0, 10)}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    // Safety net: an unexpected error anywhere (e.g. in a form renderer) is reported to the user and the
    // training buttons are re-enabled, instead of leaving the app looking busy or half-updated
    function handleUnexpectedError(err) {
        const msg = err && err.message ? err.message : String(err || 'Unknown error');
        if (msg === 'Training aborted by user.' || msg === 'Operation halted by user.') return;
        console.error('Unexpected error:', err);
        if (DOM.trainModelBtn) DOM.trainModelBtn.disabled = false;
        if (DOM.pipelineTrainModelsBtn) DOM.pipelineTrainModelsBtn.disabled = false;
        try { showSeedToast(`⚠️ Unexpected error: ${msg}. Please check your data/settings and retry.`); } catch (e) { /* ignore */ }
    }
    window.addEventListener('error', (e) => handleUnexpectedError(e.error || e.message));
    window.addEventListener('unhandledrejection', (e) => handleUnexpectedError(e.reason));

    document.addEventListener('DOMContentLoaded', () => {
        initDOM();
        setupEventListeners();
    });

})();
