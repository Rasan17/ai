/**
 * NPH ML/Neuralnet Predictor - Machine Learning & Deep Learning Engine
 * Provides Deep Neural Networks (TensorFlow.js + Pure JS Fallback), Random Forests,
 * Logistic/Softmax Regression, Ridge Regression, k-NN,
 * and Gaussian Naive Bayes with comprehensive evaluation metrics.
 * 
 * Conceived, designed and developed by Dr G Narenthiran MB ChB BSc(MedSci)(Hons) FEBNS FRCS(SN)
 */

(function (window) {
    'use strict';

    /**
     * Pseudo-Random Number Generator (Mulberry32) for deterministic reproducible modeling
     */
    function createPRNG(seed) {
        if (seed === null || seed === undefined || seed === '' || isNaN(seed)) {
            return Math.random;
        }
        let s = Math.floor(Math.abs(Number(seed))) >>> 0;
        if (s === 0) s = 1;
        return function () {
            s |= 0; s = (s + 0x6D2B79F5) | 0;
            let t = Math.imul(s ^ (s >>> 15), 1 | s);
            t = t + Math.imul(t ^ (t >>> 7), 61 | t) ^ t;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
    }

    /**
     * Yields control back to the browser between training epochs. TF.js's default yielding uses
     * requestAnimationFrame, which never fires in a background tab (training would freeze); here a
     * MessageChannel is used when the page is hidden, and setTimeout (allowing UI repaint) when visible.
     */
    function yieldToBrowser() {
        if (typeof document !== 'undefined' && document.hidden && typeof MessageChannel !== 'undefined') {
            return new Promise(resolve => {
                const ch = new MessageChannel();
                ch.port1.onmessage = () => resolve();
                ch.port2.postMessage(null);
            });
        }
        return new Promise(resolve => setTimeout(resolve, 0));
    }

    // Row-by-row outputs for any model, using a batched path when the model provides one
    function batchOutputs(model, X) {
        return typeof model.predictOutputs === 'function' ? model.predictOutputs(X) : X.map(x => model.predictSingle(x));
    }

    /**
     * Drives a step generator (one step = one tree / one epoch) asynchronously, yielding to the browser
     * every `every` steps so the UI can repaint and Force Stop can take effect. `opts.onYield` is called
     * at each yield; if it throws (e.g. user cancellation), training stops and the error propagates.
     */
    async function runStepsAsync(steps, opts = {}, every = 1) {
        let n = 0;
        for (const _ of steps) { // eslint-disable-line no-unused-vars
            if (++n % every === 0) {
                await yieldToBrowser();
                if (typeof opts.onYield === 'function') opts.onYield(n);
            }
        }
    }

    // ==========================================
    // 1. PURE JS DECISION TREE & RANDOM FOREST
    // ==========================================
    class DecisionTreeNode {
        constructor(isLeaf = false, value = null, featureIdx = null, threshold = null, left = null, right = null) {
            this.isLeaf = isLeaf;
            this.value = value; // For classification: class or prob dict; for regression: mean value
            this.featureIdx = featureIdx;
            this.threshold = threshold;
            this.left = left;
            this.right = right;
        }
    }

    class DecisionTree {
        constructor(isClassification = true, maxDepth = 6, minSamplesSplit = 2, maxFeatures = null, rng = Math.random) {
            this.isClassification = isClassification;
            this.maxDepth = maxDepth;
            this.minSamplesSplit = minSamplesSplit;
            this.maxFeatures = maxFeatures;
            this.rng = rng || Math.random;
            this.root = null;
            this.featureImportances = {};
        }

        fit(X, y) {
            const numFeatures = X[0].length;
            this.featureImportances = new Array(numFeatures).fill(0);
            this.root = this._buildTree(X, y, 0);
            // Normalize feature importances
            const sum = this.featureImportances.reduce((a, b) => a + b, 0);
            if (sum > 0) {
                this.featureImportances = this.featureImportances.map(v => v / sum);
            }
        }

        _buildTree(X, y, depth) {
            const numSamples = X.length;
            const numFeatures = X[0].length;

            if (this.isClassification) {
                const uniqueClasses = [...new Set(y)];
                if (uniqueClasses.length === 1 || depth >= this.maxDepth || numSamples < this.minSamplesSplit) {
                    const classCounts = {};
                    y.forEach(c => classCounts[c] = (classCounts[c] || 0) + 1);
                    let majority = y[0];
                    let maxCount = 0;
                    Object.keys(classCounts).forEach(c => {
                        if (classCounts[c] > maxCount) {
                            maxCount = classCounts[c];
                            majority = Number(c);
                        }
                    });
                    return new DecisionTreeNode(true, majority);
                }
            } else {
                const mean = y.reduce((a, b) => a + b, 0) / (numSamples || 1);
                if (depth >= this.maxDepth || numSamples < this.minSamplesSplit) {
                    return new DecisionTreeNode(true, mean);
                }
            }

            // Choose random subset of features
            let featureIndices = Array.from({ length: numFeatures }, (_, i) => i);
            if (this.maxFeatures && this.maxFeatures < numFeatures) {
                // Seeded Fisher-Yates shuffle (a sort() with a random comparator is not a uniform shuffle)
                for (let i = featureIndices.length - 1; i > 0; i--) {
                    const j = Math.floor(this.rng() * (i + 1));
                    [featureIndices[i], featureIndices[j]] = [featureIndices[j], featureIndices[i]];
                }
                featureIndices = featureIndices.slice(0, this.maxFeatures);
            }

            let bestGain = -Infinity;
            let bestFeature = null;
            let bestThreshold = null;
            let bestSplits = null;

            const currentScore = this.isClassification ? this._gini(y) : this._variance(y);

            featureIndices.forEach(featIdx => {
                const values = X.map(r => r[featIdx]);
                // Sample potential split thresholds
                const uniqueVals = [...new Set(values)].sort((a, b) => a - b);
                if (uniqueVals.length <= 1) return;

                const step = Math.max(1, Math.floor(uniqueVals.length / 10));
                for (let i = 0; i < uniqueVals.length - 1; i += step) {
                    const threshold = (uniqueVals[i] + uniqueVals[i + 1]) / 2;
                    const leftX = [], leftY = [], rightX = [], rightY = [];

                    for (let s = 0; s < numSamples; s++) {
                        if (X[s][featIdx] <= threshold) {
                            leftX.push(X[s]);
                            leftY.push(y[s]);
                        } else {
                            rightX.push(X[s]);
                            rightY.push(y[s]);
                        }
                    }

                    if (leftY.length === 0 || rightY.length === 0) continue;

                    let gain = 0;
                    if (this.isClassification) {
                        const leftGini = this._gini(leftY);
                        const rightGini = this._gini(rightY);
                        gain = currentScore - ((leftY.length / numSamples) * leftGini + (rightY.length / numSamples) * rightGini);
                    } else {
                        const leftVar = this._variance(leftY);
                        const rightVar = this._variance(rightY);
                        gain = currentScore - ((leftY.length / numSamples) * leftVar + (rightY.length / numSamples) * rightVar);
                    }

                    if (gain > bestGain) {
                        bestGain = gain;
                        bestFeature = featIdx;
                        bestThreshold = threshold;
                        bestSplits = { leftX, leftY, rightX, rightY };
                    }
                }
            });

            if (bestGain <= 0 || !bestSplits) {
                if (this.isClassification) {
                    const counts = {};
                    y.forEach(c => counts[c] = (counts[c] || 0) + 1);
                    let majority = y[0];
                    let maxCount = 0;
                    Object.keys(counts).forEach(c => {
                        if (counts[c] > maxCount) {
                            maxCount = counts[c];
                            majority = Number(c);
                        }
                    });
                    return new DecisionTreeNode(true, majority);
                } else {
                    return new DecisionTreeNode(true, y.reduce((a, b) => a + b, 0) / (numSamples || 1));
                }
            }

            this.featureImportances[bestFeature] += bestGain * numSamples;

            const leftChild = this._buildTree(bestSplits.leftX, bestSplits.leftY, depth + 1);
            const rightChild = this._buildTree(bestSplits.rightX, bestSplits.rightY, depth + 1);

            return new DecisionTreeNode(false, null, bestFeature, bestThreshold, leftChild, rightChild);
        }

        _gini(y) {
            const counts = {};
            y.forEach(c => counts[c] = (counts[c] || 0) + 1);
            let sumSq = 0;
            const n = y.length;
            Object.values(counts).forEach(cnt => {
                const p = cnt / n;
                sumSq += p * p;
            });
            return 1 - sumSq;
        }

        _variance(y) {
            const n = y.length;
            if (n === 0) return 0;
            const mean = y.reduce((a, b) => a + b, 0) / n;
            return y.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / n;
        }

        predictSample(x) {
            let node = this.root;
            while (!node.isLeaf) {
                if (x[node.featureIdx] <= node.threshold) {
                    node = node.left;
                } else {
                    node = node.right;
                }
            }
            return node.value;
        }
    }

    class RandomForest {
        constructor(isClassification = true, numTrees = 25, maxDepth = 5, minSamplesSplit = 2, featureRatio = 'sqrt', seed = null) {
            this.isClassification = isClassification;
            this.numTrees = numTrees;
            this.maxDepth = maxDepth;
            this.minSamplesSplit = minSamplesSplit;
            this.featureRatio = featureRatio;
            this.seed = seed !== null && seed !== undefined && seed !== '' ? Number(seed) : null;
            this.rng = createPRNG(this.seed);
            this.trees = [];
            this.featureImportances = [];
        }

        fit(X, y) {
            for (const _ of this._fitSteps(X, y)) { /* synchronous: run every tree */ } // eslint-disable-line no-unused-vars
        }

        // Same fit, but yields to the browser after every tree (keeps the UI and Force Stop responsive)
        async fitAsync(X, y, opts = {}) {
            await runStepsAsync(this._fitSteps(X, y), opts, 1);
        }

        *_fitSteps(X, y) {
            const numSamples = X.length;
            const numFeatures = X[0].length;

            let maxFeatures = Math.max(1, Math.floor(this.isClassification ? Math.sqrt(numFeatures) : numFeatures / 3));
            if (this.featureRatio === 'all') {
                maxFeatures = numFeatures;
            } else if (this.featureRatio === 'log2') {
                maxFeatures = Math.max(1, Math.floor(Math.log2(numFeatures)));
            } else if (this.featureRatio === '0.5') {
                maxFeatures = Math.max(1, Math.floor(numFeatures * 0.5));
            } else if (this.featureRatio === '0.75') {
                maxFeatures = Math.max(1, Math.floor(numFeatures * 0.75));
            }

            this.trees = [];
            const aggregatedImportances = new Array(numFeatures).fill(0);

            for (let t = 0; t < this.numTrees; t++) {
                // Bootstrap sample with replacement using seeded PRNG
                const sampleX = [];
                const sampleY = [];
                for (let i = 0; i < numSamples; i++) {
                    const randIdx = Math.floor(this.rng() * numSamples);
                    sampleX.push(X[randIdx]);
                    sampleY.push(y[randIdx]);
                }

                const tree = new DecisionTree(this.isClassification, this.maxDepth, this.minSamplesSplit, maxFeatures, this.rng);
                tree.fit(sampleX, sampleY);
                this.trees.push(tree);

                tree.featureImportances.forEach((imp, fIdx) => {
                    aggregatedImportances[fIdx] += imp;
                });
                yield t;
            }

            const total = aggregatedImportances.reduce((a, b) => a + b, 0);
            this.featureImportances = aggregatedImportances.map(v => total > 0 ? Number((v / total).toFixed(4)) : 0);
        }

        predict(X) {
            return X.map(x => this.predictSingle(x).pred);
        }

        predictSingle(x) {
            const votes = this.trees.map(tree => tree.predictSample(x));
            if (this.isClassification) {
                const counts = {};
                votes.forEach(v => counts[v] = (counts[v] || 0) + 1);
                let majority = votes[0];
                let maxCount = 0;
                Object.keys(counts).forEach(v => {
                    if (counts[v] > maxCount) {
                        maxCount = counts[v];
                        majority = Number(v);
                    }
                });
                const probs = {};
                Object.keys(counts).forEach(k => probs[k] = counts[k] / votes.length);
                return { pred: majority, probabilities: probs, confidence: maxCount / votes.length, probabilityType: 'vote' };
            } else {
                const mean = votes.reduce((a, b) => a + b, 0) / votes.length;
                return { pred: Number(mean.toFixed(4)), confidence: null };
            }
        }
    }

    // ==========================================
    // 2. LOGISTIC / SOFTMAX & RIDGE REGRESSION
    // ==========================================
    class GeneralizedLinearModel {
        constructor(isClassification = true, numClasses = 2, learningRate = 0.05, epochs = 150, lambdaL2 = 0.01, solver = 'momentum', seed = null) {
            this.isClassification = isClassification;
            this.numClasses = numClasses;
            this.lr = learningRate;
            this.epochs = epochs;
            this.lambdaL2 = lambdaL2;
            this.solver = solver;
            this.seed = seed !== null && seed !== undefined && seed !== '' ? Number(seed) : null;
            this.rng = createPRNG(this.seed);
            this.weights = null; // matrix or vector
            this.bias = null;
        }

        fit(X, y) {
            for (const _ of this._fitSteps(X, y)) { /* synchronous: run every epoch */ } // eslint-disable-line no-unused-vars
        }

        // Same fit, but yields to the browser every 10 epochs (keeps the UI and Force Stop responsive)
        async fitAsync(X, y, opts = {}) {
            await runStepsAsync(this._fitSteps(X, y), opts, 10);
        }

        *_fitSteps(X, y) {
            const n = X.length;
            const p = X[0].length;
            const momentumBeta = this.solver === 'momentum' ? 0.85 : 0.0;

            if (this.isClassification) {
                // Multinomial Softmax or Binary Sigmoid
                this.weights = Array.from({ length: this.numClasses }, () => new Array(p).fill(0));
                this.bias = new Array(this.numClasses).fill(0);
                const velW = Array.from({ length: this.numClasses }, () => new Array(p).fill(0));
                const velB = new Array(this.numClasses).fill(0);

                for (let epoch = 0; epoch < this.epochs; epoch++) {
                    const gradW = Array.from({ length: this.numClasses }, () => new Array(p).fill(0));
                    const gradB = new Array(this.numClasses).fill(0);

                    for (let i = 0; i < n; i++) {
                        const logits = new Array(this.numClasses).fill(0);
                        for (let c = 0; c < this.numClasses; c++) {
                            let dot = this.bias[c];
                            for (let j = 0; j < p; j++) {
                                dot += this.weights[c][j] * X[i][j];
                            }
                            logits[c] = dot;
                        }

                        // Softmax
                        const maxLogit = Math.max(...logits);
                        const exp = logits.map(l => Math.exp(l - maxLogit));
                        const sumExp = exp.reduce((a, b) => a + b, 0);
                        const probs = exp.map(e => e / (sumExp || 1));

                        const trueClass = y[i];
                        for (let c = 0; c < this.numClasses; c++) {
                            const err = probs[c] - (c === trueClass ? 1 : 0);
                            gradB[c] += err;
                            for (let j = 0; j < p; j++) {
                                gradW[c][j] += err * X[i][j];
                            }
                        }
                    }

                    // Update with momentum and regularization
                    for (let c = 0; c < this.numClasses; c++) {
                        const gB = gradB[c] / n;
                        velB[c] = momentumBeta * velB[c] + this.lr * gB;
                        this.bias[c] -= velB[c];

                        for (let j = 0; j < p; j++) {
                            const reg = this.lambdaL2 * this.weights[c][j];
                            const gW = (gradW[c][j] / n) + reg;
                            velW[c][j] = momentumBeta * velW[c][j] + this.lr * gW;
                            this.weights[c][j] -= velW[c][j];
                        }
                    }
                    this._checkFinite(epoch);
                    yield epoch;
                }
            } else {
                // Ridge Linear Regression via Gradient Descent with momentum.
                // Objective: (1/2n)·Σ(pred − y)² + (λ/2)·‖w‖² (intercept unpenalised), whose gradient is
                // (1/n)·Σ err·x + λw, exactly as computed below. This is equivalent to scikit-learn
                // Ridge(alpha = n·λ) (checked in tests/reference_check.py); the learning rate applies to it.
                this.weights = new Array(p).fill(0);
                this.bias = 0;
                const velW = new Array(p).fill(0);
                let velB = 0;

                for (let epoch = 0; epoch < this.epochs; epoch++) {
                    let gradB = 0;
                    const gradW = new Array(p).fill(0);

                    for (let i = 0; i < n; i++) {
                        let pred = this.bias;
                        for (let j = 0; j < p; j++) {
                            pred += this.weights[j] * X[i][j];
                        }
                        const err = pred - y[i];
                        gradB += err;
                        for (let j = 0; j < p; j++) {
                            gradW[j] += err * X[i][j];
                        }
                    }

                    const gB = gradB / n;
                    velB = momentumBeta * velB + this.lr * gB;
                    this.bias -= velB;

                    for (let j = 0; j < p; j++) {
                        const reg = this.lambdaL2 * this.weights[j];
                        const gW = (gradW[j] / n) + reg;
                        velW[j] = momentumBeta * velW[j] + this.lr * gW;
                        this.weights[j] -= velW[j];
                    }
                    this._checkFinite(epoch);
                    yield epoch;
                }
            }
        }

        _checkFinite(epoch) {
            const flat = this.isClassification ? [...this.bias, ...this.weights.flat()] : [this.bias, ...this.weights];
            if (flat.some(v => !Number.isFinite(v))) {
                throw new Error(`Logistic/Ridge regression diverged at iteration ${epoch + 1} (coefficients became non-finite). Reduce the learning rate or increase the L2 penalty.`);
            }
        }

        predictSingle(x) {
            const p = x.length;
            if (this.isClassification) {
                const logits = new Array(this.numClasses).fill(0);
                for (let c = 0; c < this.numClasses; c++) {
                    let dot = this.bias[c];
                    for (let j = 0; j < p; j++) {
                        dot += this.weights[c][j] * x[j];
                    }
                    logits[c] = dot;
                }
                const maxLogit = Math.max(...logits);
                const exp = logits.map(l => Math.exp(l - maxLogit));
                const sumExp = exp.reduce((a, b) => a + b, 0);
                const probs = exp.map(e => e / (sumExp || 1));

                let bestClass = 0;
                let maxP = 0;
                const probMap = {};
                probs.forEach((prob, c) => {
                    probMap[c] = Number(prob.toFixed(4));
                    if (prob > maxP) {
                        maxP = prob;
                        bestClass = c;
                    }
                });

                return { pred: bestClass, probabilities: probMap, confidence: Number(maxP.toFixed(4)), probabilityType: 'model' };
            } else {
                let pred = this.bias;
                for (let j = 0; j < p; j++) {
                    pred += this.weights[j] * x[j];
                }
                return { pred: Number(pred.toFixed(4)), confidence: null };
            }
        }

        predict(X) {
            return X.map(x => this.predictSingle(x).pred);
        }
    }

    // ==========================================
    // 3. k-NEAREST NEIGHBORS (k-NN)
    // ==========================================
    class KNNModel {
        constructor(isClassification = true, k = 5, metric = 'euclidean', weighting = 'uniform', seed = null) {
            this.isClassification = isClassification;
            this.k = k;
            this.metric = metric; // 'euclidean', 'manhattan', 'chebyshev'
            this.weighting = weighting; // 'uniform', 'distance'
            this.seed = seed !== null && seed !== undefined && seed !== '' ? Number(seed) : null;
            this.trainX = [];
            this.trainY = [];
        }

        fit(X, y) {
            this.trainX = X;
            this.trainY = y;
            // Seeded tie-break key per training case: equidistant neighbours are ordered by this key
            // (otherwise ties always favour rows earlier in the CSV); identical seed -> identical result
            const rng = createPRNG(this.seed);
            this.tieKeys = X.map(() => rng());
        }

        predictSingle(x) {
            const distances = this.trainX.map((tx, idx) => {
                let d = 0;
                if (this.metric === 'manhattan') {
                    for (let j = 0; j < x.length; j++) {
                        d += Math.abs(x[j] - tx[j]);
                    }
                } else if (this.metric === 'chebyshev') {
                    for (let j = 0; j < x.length; j++) {
                        d = Math.max(d, Math.abs(x[j] - tx[j]));
                    }
                } else {
                    // euclidean
                    let sumSq = 0;
                    for (let j = 0; j < x.length; j++) {
                        const diff = x[j] - tx[j];
                        sumSq += diff * diff;
                    }
                    d = Math.sqrt(sumSq);
                }
                return { dist: d, label: this.trainY[idx], tie: this.tieKeys ? this.tieKeys[idx] : idx };
            });

            // Top-k selection (k is small) instead of sorting every training case: same order as sorting by
            // (distance, seeded tie key), at O(n·k) rather than O(n log n) per prediction
            const kEff = Math.min(this.k, distances.length);
            const kNearest = [];
            const before = (a, b) => (a.dist - b.dist) < 0 || (a.dist === b.dist && a.tie < b.tie);
            for (let i = 0; i < distances.length; i++) {
                const cand = distances[i];
                if (kNearest.length === kEff && !before(cand, kNearest[kEff - 1])) continue;
                let pos = kNearest.length;
                while (pos > 0 && before(cand, kNearest[pos - 1])) pos--;
                kNearest.splice(pos, 0, cand);
                if (kNearest.length > kEff) kNearest.pop();
            }

            if (this.isClassification) {
                const weightedCounts = {};
                let totalWeight = 0;
                kNearest.forEach(n => {
                    const w = this.weighting === 'distance' ? (1.0 / (n.dist + 1e-5)) : 1.0;
                    weightedCounts[n.label] = (weightedCounts[n.label] || 0) + w;
                    totalWeight += w;
                });

                let majority = kNearest[0].label;
                let maxW = 0;
                Object.keys(weightedCounts).forEach(l => {
                    if (weightedCounts[l] > maxW) {
                        maxW = weightedCounts[l];
                        majority = Number(l);
                    }
                });

                const probs = {};
                Object.keys(weightedCounts).forEach(k => {
                    probs[k] = Number((weightedCounts[k] / (totalWeight || 1)).toFixed(4));
                });
                return {
                    pred: majority,
                    probabilities: probs,
                    confidence: Number((maxW / (totalWeight || 1)).toFixed(4)),
                    probabilityType: 'neighbours'
                };
            } else {
                let weightedSum = 0;
                let totalWeight = 0;
                kNearest.forEach(n => {
                    const w = this.weighting === 'distance' ? (1.0 / (n.dist + 1e-5)) : 1.0;
                    weightedSum += n.label * w;
                    totalWeight += w;
                });
                const mean = weightedSum / (totalWeight || 1);
                return { pred: Number(mean.toFixed(4)), confidence: null };
            }
        }

        predict(X) {
            return X.map(x => this.predictSingle(x).pred);
        }
    }

    // ==========================================
    // 4. NAIVE BAYES (mixed likelihoods)
    // Features that take only the values 0/1 in the training data (one-hot category indicators,
    // binary flags) use a Bernoulli likelihood with Laplace smoothing; all other features use a
    // Gaussian likelihood. (Class name kept for backwards compatibility.)
    // ==========================================
    class GaussianNaiveBayes {
        constructor(varSmoothing = 1e-4, priorMode = 'empirical', seed = null) {
            this.varSmoothing = varSmoothing; // 1e-9, 1e-4, 1e-2, 0.1
            this.priorMode = priorMode; // 'empirical', 'uniform', 'laplace'
            this.seed = seed !== null && seed !== undefined && seed !== '' ? Number(seed) : null;
            this.classes = [];
            this.priors = {};
            this.means = {};
            this.variances = {};
            this.bernoulliP = {};
            this.isBinaryFeature = [];
        }

        fit(X, y) {
            this.classes = [...new Set(y)];
            const n = X.length;
            const p = X[0].length;
            const numK = this.classes.length;
            this.priors = {};
            this.means = {};
            this.variances = {};
            this.bernoulliP = {};
            this.isBinaryFeature = Array.from({ length: p }, (_, j) => X.every(r => r[j] === 0 || r[j] === 1));

            this.classes.forEach(c => {
                const classX = X.filter((_, idx) => y[idx] === c);
                const count = classX.length;

                if (this.priorMode === 'uniform') {
                    this.priors[c] = 1.0 / (numK || 1);
                } else if (this.priorMode === 'laplace') {
                    this.priors[c] = (count + 1) / (n + numK);
                } else {
                    // empirical
                    this.priors[c] = count / (n || 1);
                }

                this.means[c] = new Array(p).fill(0);
                this.variances[c] = new Array(p).fill(this.varSmoothing);
                this.bernoulliP[c] = new Array(p).fill(0.5);

                for (let j = 0; j < p; j++) {
                    const vals = classX.map(r => r[j]);
                    if (this.isBinaryFeature[j]) {
                        // Laplace-smoothed P(feature = 1 | class)
                        const ones = vals.reduce((a, b) => a + b, 0);
                        this.bernoulliP[c][j] = (ones + 1) / (vals.length + 2);
                        continue;
                    }
                    const mean = vals.reduce((a, b) => a + b, 0) / (vals.length || 1);
                    this.means[c][j] = mean;
                    const variance = vals.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (vals.length || 1);
                    this.variances[c][j] = Math.max(variance, this.varSmoothing);
                }
            });
        }

        predictSingle(x) {
            const logProbs = {};
            const p = x.length;

            this.classes.forEach(c => {
                let logP = Math.log(this.priors[c] || 1e-6);
                for (let j = 0; j < p; j++) {
                    if (this.isBinaryFeature[j]) {
                        const pj = this.bernoulliP[c][j];
                        logP += x[j] >= 0.5 ? Math.log(pj) : Math.log(1 - pj);
                        continue;
                    }
                    const mean = this.means[c][j];
                    const v = this.variances[c][j];
                    const diff = x[j] - mean;
                    logP += -0.5 * Math.log(2 * Math.PI * v) - (diff * diff) / (2 * v);
                }
                logProbs[c] = logP;
            });

            const maxLogP = Math.max(...Object.values(logProbs));
            const exps = {};
            let sumExp = 0;
            Object.keys(logProbs).forEach(c => {
                const exp = Math.exp(logProbs[c] - maxLogP);
                exps[c] = exp;
                sumExp += exp;
            });

            let bestClass = this.classes[0];
            let maxProb = 0;
            const probMap = {};
            Object.keys(exps).forEach(c => {
                const prob = exps[c] / (sumExp || 1);
                probMap[c] = Number(prob.toFixed(4));
                if (prob > maxProb) {
                    maxProb = prob;
                    bestClass = Number(c);
                }
            });

            return { pred: bestClass, probabilities: probMap, confidence: Number(maxProb.toFixed(4)), probabilityType: 'model' };
        }

        predict(X) {
            return X.map(x => this.predictSingle(x).pred);
        }
    }

    // ==========================================
    // 5. TENSORFLOW.JS / NATIVE DEEP NEURAL NETWORK
    // ==========================================
    class DeepNeuralNetwork {
        constructor(isClassification = true, numClasses = 2, config = {}) {
            this.isClassification = isClassification;
            this.numClasses = numClasses;
            this.hiddenLayers = config.hiddenLayers || [64, 32];
            this.activation = config.activation || 'relu';
            this.learningRate = config.learningRate || 0.005;
            this.epochs = config.epochs || 50;
            this.batchSize = config.batchSize || 16;
            this.dropoutRate = config.dropoutRate !== undefined ? config.dropoutRate : 0.1;
            this.optimizerName = config.optimizer || 'adam';
            this.l2Reg = config.l2Reg !== undefined ? config.l2Reg : 0.0001;
            this.seed = config.seed !== null && config.seed !== undefined && config.seed !== '' ? Number(config.seed) : null;
            this.rng = createPRNG(this.seed);
            this.tfModel = null;
            this.fallbackGLM = null;
            this.history = { loss: [], valLoss: [], accuracy: [], valAccuracy: [] };
            // Internal validation split (carved from the TRAINING data) used for early stopping
            this.earlyStopping = config.earlyStopping !== false;
            this.validationFraction = config.validationFraction !== undefined ? config.validationFraction : 0.15;
            this.patience = config.patience || Math.max(5, Math.round(this.epochs * 0.2));
            this.trainingInfo = null;
        }

        /**
         * Splits training data into fit / validation parts (stratified by class for classification),
         * using the model's seeded RNG. Returns null when there are too few cases to hold any out.
         */
        _carveValidationSplit(X, Y) {
            const n = X.length;
            if (!this.earlyStopping || this.validationFraction <= 0 || n < 20) return null;
            const idx = Array.from({ length: n }, (_, i) => i);
            for (let i = n - 1; i > 0; i--) {
                const j = Math.floor(this.rng() * (i + 1));
                [idx[i], idx[j]] = [idx[j], idx[i]];
            }
            const fitIdx = [];
            const valIdx = [];
            if (this.isClassification) {
                const byClass = {};
                idx.forEach(i => { (byClass[Y[i]] = byClass[Y[i]] || []).push(i); });
                Object.values(byClass).forEach(list => {
                    // Keep at least 2 cases of every class for fitting
                    const nVal = list.length >= 4 ? Math.max(1, Math.round(list.length * this.validationFraction)) : 0;
                    valIdx.push(...list.slice(0, nVal));
                    fitIdx.push(...list.slice(nVal));
                });
            } else {
                const nVal = Math.max(1, Math.round(n * this.validationFraction));
                valIdx.push(...idx.slice(0, nVal));
                fitIdx.push(...idx.slice(nVal));
            }
            if (valIdx.length < 3) return null;
            return {
                fitX: fitIdx.map(i => X[i]), fitY: fitIdx.map(i => Y[i]),
                valX: valIdx.map(i => X[i]), valY: valIdx.map(i => Y[i])
            };
        }

        /**
         * validX / validY: optional explicit validation data. It must come from the training data,
         * NEVER the test set. When omitted, an internal stratified validation split is carved from
         * trainX (see validationFraction) and used for early stopping with best-weight restoration.
         */
        async fit(trainX, trainY, validX = null, validY = null, onEpochCallback = null) {
            const hasTF = typeof window.tf !== 'undefined' && window.tf !== null;
            const inputDim = trainX[0].length;
            this.usedFallback = false;
            this.fallbackReason = null;
            this.trainingInfo = null;
            // Fresh seeded RNG per fit so refits with the same seed reproduce the same shuffles
            this.rng = createPRNG(this.seed);

            const allTrainX = trainX;
            const allTrainY = trainY;
            if (!(validX && validY && validX.length > 0)) {
                const carved = this._carveValidationSplit(trainX, trainY);
                if (carved) {
                    trainX = carved.fitX;
                    trainY = carved.fitY;
                    validX = carved.valX;
                    validY = carved.valY;
                } else {
                    validX = null;
                    validY = null;
                }
            }

            if (hasTF) {
                // Dispose existing model instance (and its optimizer state) if retraining
                this.dispose();

                let model = null;
                let valXTensor = null;
                let valYTensor = null;
                let xAll = null;
                let yAll = null;
                let bestWeights = null;
                const disposeBest = () => {
                    if (bestWeights) bestWeights.forEach(w => { try { w.dispose(); } catch (e) {} });
                    bestWeights = null;
                };

                try {
                    const tf = window.tf;
                    model = tf.sequential();

                    // Optional L2 kernel regularizer
                    const kernelReg = (this.l2Reg && this.l2Reg > 0 && typeof tf.regularizers !== 'undefined' && tf.regularizers.l2)
                        ? tf.regularizers.l2({ l2: this.l2Reg })
                        : null;

                    // Seed-based weight initializer for reproducibility (a different derived seed per layer)
                    const hasSeededInit = this.seed !== null && typeof tf.initializers !== 'undefined' && tf.initializers.glorotUniform;
                    const kernelInitFor = (layerIdx) => (hasSeededInit
                        ? tf.initializers.glorotUniform({ seed: this.seed + layerIdx * 7919 })
                        : 'glorotUniform');

                    // Input + First Hidden Layer
                    const firstDense = {
                        inputShape: [inputDim],
                        units: this.hiddenLayers[0] || 64,
                        activation: this.activation,
                        kernelInitializer: kernelInitFor(0)
                    };
                    if (kernelReg) firstDense.kernelRegularizer = kernelReg;
                    model.add(tf.layers.dense(firstDense));

                    if (this.dropoutRate > 0) {
                        const dropCfg = { rate: this.dropoutRate };
                        if (this.seed !== null) dropCfg.seed = this.seed;
                        model.add(tf.layers.dropout(dropCfg));
                    }

                    // Subsequent Hidden Layers
                    for (let l = 1; l < this.hiddenLayers.length; l++) {
                        const units = this.hiddenLayers[l];
                        if (!units || units <= 0) continue;
                        const nextDense = {
                            units: units,
                            activation: this.activation,
                            kernelInitializer: kernelInitFor(l)
                        };
                        if (kernelReg) nextDense.kernelRegularizer = kernelReg;
                        model.add(tf.layers.dense(nextDense));
                        if (this.dropoutRate > 0) {
                            const dropCfg = { rate: this.dropoutRate };
                            if (this.seed !== null) dropCfg.seed = this.seed + l;
                            model.add(tf.layers.dropout(dropCfg));
                        }
                    }

                    // Output Layer
                    const outIdx = this.hiddenLayers.length;
                    if (this.isClassification) {
                        model.add(tf.layers.dense({
                            units: this.numClasses,
                            activation: 'softmax',
                            kernelInitializer: kernelInitFor(outIdx)
                        }));
                    } else {
                        model.add(tf.layers.dense({ units: 1, kernelInitializer: kernelInitFor(outIdx) }));
                    }

                    // Optimizer
                    let opt;
                    if (this.optimizerName === 'sgd') {
                        opt = tf.train.sgd(this.learningRate);
                    } else if (this.optimizerName === 'rmsprop') {
                        opt = tf.train.rmsprop(this.learningRate);
                    } else {
                        opt = tf.train.adam(this.learningRate);
                    }

                    model.compile({
                        optimizer: opt,
                        loss: this.isClassification ? 'sparseCategoricalCrossentropy' : 'meanSquaredError',
                        metrics: this.isClassification ? ['accuracy'] : ['mse']
                    });

                    // Validation data (from the training data only) drives early stopping
                    let validationData = undefined;
                    if (validX && validY && validX.length > 0) {
                        valXTensor = tf.tensor2d(validX, [validX.length, inputDim]);
                        // Class indices as float32: TF.js 4.x's accuracy metric fails on int32 labels
                        valYTensor = this.isClassification ?
                            tf.tensor1d(validY, 'float32') :
                            tf.tensor2d(validY.map(v => [v]), [validY.length, 1]);
                        validationData = [valXTensor, valYTensor];
                    }

                    const history = { loss: [], valLoss: [], accuracy: [], valAccuracy: [] };
                    const n = trainX.length;
                    const batchSize = Math.min(this.batchSize, n);
                    let bestValLoss = Infinity;
                    let bestEpoch = 0;
                    let epochsWithoutImprovement = 0;
                    let stoppedEpoch = this.epochs;
                    let earlyStopped = false;

                    // Training tensors are built once; each epoch reorders them with tf.gather
                    xAll = tf.tensor2d(trainX, [n, inputDim]);
                    yAll = this.isClassification ?
                        tf.tensor1d(trainY, 'float32') :
                        tf.tensor2d(trainY.map(v => [v]), [n, 1]);

                    // Epoch loop with our own seeded shuffle (TF.js's built-in shuffle uses an unseeded RNG)
                    for (let epoch = 0; epoch < this.epochs; epoch++) {
                        const perm = Array.from({ length: n }, (_, i) => i);
                        for (let i = n - 1; i > 0; i--) {
                            const j = Math.floor(this.rng() * (i + 1));
                            [perm[i], perm[j]] = [perm[j], perm[i]];
                        }
                        const permTensor = tf.tensor1d(perm, 'int32');
                        const xTensor = tf.gather(xAll, permTensor);
                        const yTensor = tf.gather(yAll, permTensor);
                        permTensor.dispose();

                        let h;
                        try {
                            h = await model.fit(xTensor, yTensor, {
                                epochs: 1,
                                batchSize,
                                validationData,
                                shuffle: false,
                                yieldEvery: 'never' // we yield per epoch ourselves (works in background tabs)
                            });
                        } finally {
                            xTensor.dispose();
                            yTensor.dispose();
                        }

                        const H = h.history;
                        const logs = {
                            loss: H.loss ? H.loss[0] : NaN,
                            val_loss: H.val_loss ? H.val_loss[0] : undefined,
                            acc: H.acc ? H.acc[0] : (H.accuracy ? H.accuracy[0] : undefined),
                            val_acc: H.val_acc ? H.val_acc[0] : (H.val_accuracy ? H.val_accuracy[0] : undefined)
                        };
                        if (!Number.isFinite(logs.loss)) {
                            throw new Error(`Neural network training diverged at epoch ${epoch + 1} (loss is not finite). Reduce the learning rate.`);
                        }

                        history.loss.push(Number(logs.loss.toFixed(4)));
                        if (logs.val_loss !== undefined) history.valLoss.push(Number(logs.val_loss.toFixed(4)));
                        if (logs.acc !== undefined) history.accuracy.push(Number(logs.acc.toFixed(4)));
                        if (logs.val_acc !== undefined) history.valAccuracy.push(Number(logs.val_acc.toFixed(4)));

                        if (onEpochCallback) {
                            try {
                                onEpochCallback(epoch + 1, this.epochs, logs);
                            } catch (cbErr) {
                                // e.g. user cancellation via the watchdog: propagate, do not fall back
                                if (cbErr && typeof cbErr === 'object') cbErr.isCallbackError = true;
                                throw cbErr;
                            }
                        }
                        // Early stopping on internal validation loss, keeping the best weights
                        if (validationData && this.earlyStopping && Number.isFinite(logs.val_loss)) {
                            if (logs.val_loss < bestValLoss - 1e-6) {
                                bestValLoss = logs.val_loss;
                                bestEpoch = epoch + 1;
                                epochsWithoutImprovement = 0;
                                disposeBest();
                                bestWeights = model.getWeights().map(w => w.clone());
                            } else if (++epochsWithoutImprovement >= this.patience) {
                                stoppedEpoch = epoch + 1;
                                earlyStopped = true;
                                await yieldToBrowser();
                                break;
                            }
                        }
                        await yieldToBrowser();
                    }

                    if (bestWeights) {
                        model.setWeights(bestWeights);
                        disposeBest();
                    }
                    this.trainingInfo = {
                        fitCases: trainX.length,
                        validationCases: validationData ? validX.length : 0,
                        earlyStopping: Boolean(validationData && this.earlyStopping),
                        earlyStopped,
                        bestEpoch: bestEpoch || null,
                        stoppedEpoch,
                        maxEpochs: this.epochs
                    };

                    this.tfModel = model;
                    model = null; // ownership transferred
                    this.history = history;
                    return;
                } catch (e) {
                    if (e && e.isCallbackError) throw e;
                    console.warn("TensorFlow.js training exception, using native fallback:", e);
                    this.fallbackReason = e && e.message ? e.message : String(e);
                } finally {
                    // Guaranteed cleanup of intermediate tensors and of a partially-built model
                    disposeBest();
                    if (xAll) { try { xAll.dispose(); } catch (e) {} }
                    if (yAll) { try { yAll.dispose(); } catch (e) {} }
                    if (model) {
                        const opt = model.optimizer;
                        try { model.dispose(); } catch (e) {}
                        try { if (opt && typeof opt.dispose === 'function') opt.dispose(); } catch (e) {}
                    }
                    if (valXTensor) { try { valXTensor.dispose(); } catch (e) {} }
                    if (valYTensor) { try { valYTensor.dispose(); } catch (e) {} }
                }
            } else {
                this.fallbackReason = 'TensorFlow.js is not available';
            }

            // Fallback: native (single-layer) logistic / softmax / ridge model.
            // This is NOT a neural network; callers can detect it via `usedFallback`.
            console.warn(`Deep Neural Network unavailable (${this.fallbackReason}); trained a linear fallback model instead.`);
            this.usedFallback = true;
            this.fallbackGLM = new GeneralizedLinearModel(
                this.isClassification,
                this.numClasses,
                0.05,
                Math.max(150, this.epochs),
                this.l2Reg,
                'momentum',
                this.seed
            );
            // The fallback has no early stopping, so it uses all of the training data
            this.fallbackGLM.fit(allTrainX, allTrainY);
            // No epoch-level training history exists for the fallback model
            this.history = { loss: [], valLoss: [], accuracy: [], valAccuracy: [] };
        }

        predictSingle(x) {
            return this.predictOutputs([x])[0];
        }

        // Batched prediction: one tensor and one GPU->CPU read for all rows (per-row dataSync() calls
        // stall the browser on large evaluation or batch files)
        predictOutputs(X) {
            if (this.tfModel) {
                if (X.length === 0) return [];
                const tf = window.tf;
                const width = this.isClassification ? this.numClasses : 1;
                const data = tf.tidy(() => this.tfModel.predict(tf.tensor2d(X, [X.length, X[0].length])).dataSync());
                return X.map((_, r) => {
                    const row = data.subarray(r * width, (r + 1) * width);
                    if (!this.isClassification) {
                        return { pred: Number(row[0].toFixed(4)), confidence: null };
                    }
                    let bestClass = 0;
                    let maxP = 0;
                    const probMap = {};
                    for (let c = 0; c < this.numClasses; c++) {
                        const prob = row[c];
                        probMap[c] = Number(prob.toFixed(4));
                        if (prob > maxP) {
                            maxP = prob;
                            bestClass = c;
                        }
                    }
                    return { pred: bestClass, probabilities: probMap, confidence: Number(maxP.toFixed(4)), probabilityType: 'model' };
                });
            } else if (this.fallbackGLM) {
                return X.map(x => this.fallbackGLM.predictSingle(x));
            }
            throw new Error("Model is not trained yet.");
        }

        predict(X) {
            return this.predictOutputs(X).map(o => o.pred);
        }

        // Frees the TensorFlow.js (GPU/WebGL) memory held by this model. model.dispose() does NOT free the
        // optimizer's state (e.g. Adam's moment estimates), so the optimizer is disposed explicitly too.
        dispose() {
            if (this.tfModel) {
                const opt = this.tfModel.optimizer;
                try { this.tfModel.dispose(); } catch (e) { /* ignore */ }
                try { if (opt && typeof opt.dispose === 'function') opt.dispose(); } catch (e) { /* ignore */ }
                this.tfModel = null;
            }
        }
    }

    // ==========================================
    // 6. ENSEMBLE SUPER LEARNER (VOTING & BLENDING)
    // ==========================================
    class EnsembleModel {
        constructor(isClassification, numClasses = 2, config = {}) {
            this.isClassification = isClassification;
            this.numClasses = numClasses;
            this.voting = config.voting || 'soft'; // 'soft', 'weighted', 'hard'
            this.selectedBaseIds = config.baseModels && config.baseModels.length > 0
                ? config.baseModels
                : ['nn', 'rf', 'glm'];
            this.seed = config.seed !== null && config.seed !== undefined && config.seed !== '' ? Number(config.seed) : null;
            this.rng = createPRNG(this.seed);
            this.models = [];
            this.weights = [];
            this.baseMetrics = [];
            this.userHyperparams = config;
        }

        async _trainBaseModels(trainX, trainY, onProgress, phaseLabel = '') {
            const models = [];
            const baseIds = this.selectedBaseIds.filter(mId => this.isClassification || mId !== 'nb'); // NB is classification only
            const totalBases = baseIds.length;
            const capacity = this.userHyperparams.capacity || 'balanced';

            for (let idx = 0; idx < totalBases; idx++) {
                const mId = baseIds[idx];

                if (onProgress) {
                    onProgress(idx + 1, totalBases, {
                        stage: `${phaseLabel}Training Ensemble Base Model ${idx + 1}/${totalBases}: ${mId.toUpperCase()}...`,
                        baseId: mId
                    });
                }

                const subSeed = this.seed !== null ? (this.seed + (idx + 1) * 1007) : null;
                let subModel = null;
                if (mId === 'nn') {
                    const epochs = capacity === 'fast' ? 20 : (capacity === 'deep' ? 60 : (this.userHyperparams.nnEpochs || 35));
                    const hidden = capacity === 'fast' ? [32, 16] : (capacity === 'deep' ? [64, 48, 24] : [48, 24]);
                    const nnParams = {
                        epochs: epochs,
                        learningRate: this.userHyperparams.nnLr || 0.005,
                        hiddenLayers: hidden,
                        dropoutRate: 0.1,
                        activation: 'relu',
                        optimizer: 'adam',
                        batchSize: 16,
                        l2Reg: 0.0001,
                        seed: subSeed
                    };
                    subModel = new DeepNeuralNetwork(this.isClassification, this.numClasses, nnParams);
                    await subModel.fit(trainX, trainY, null, null, (ep, tot, logs) => {
                        if (onProgress) {
                            onProgress(idx + 1, totalBases, {
                                stage: `${phaseLabel}Base ${idx + 1}/${totalBases} (NN Epoch ${ep}/${tot} - Loss: ${logs.loss.toFixed(4)})`,
                                loss: logs.loss
                            });
                        }
                    });
                } else if (mId === 'rf') {
                    const trees = capacity === 'fast' ? 15 : (capacity === 'deep' ? 50 : (this.userHyperparams.rfTrees || 30));
                    const depth = capacity === 'fast' ? 4 : (capacity === 'deep' ? 8 : (this.userHyperparams.rfDepth || 6));
                    subModel = new RandomForest(this.isClassification, trees, depth, 2, 'sqrt', subSeed);
                    await subModel.fitAsync(trainX, trainY, {
                        // Re-report progress at each yield so a cancelling onProgress can stop training here
                        onYield: (t) => onProgress && onProgress(idx + 1, totalBases, { stage: `${phaseLabel}Base ${idx + 1}/${totalBases} (RF tree ${t}/${trees})` })
                    });
                } else if (mId === 'glm') {
                    const epochs = capacity === 'fast' ? 80 : (capacity === 'deep' ? 250 : 150);
                    subModel = new GeneralizedLinearModel(this.isClassification, this.numClasses, 0.05, epochs, 0.01, 'momentum', subSeed);
                    await subModel.fitAsync(trainX, trainY, {
                        onYield: (ep) => onProgress && onProgress(idx + 1, totalBases, { stage: `${phaseLabel}Base ${idx + 1}/${totalBases} (GLM iteration ${ep}/${epochs})` })
                    });
                } else if (mId === 'knn') {
                    const kVal = capacity === 'fast' ? 3 : (capacity === 'deep' ? 7 : (this.userHyperparams.knnK || 5));
                    subModel = new KNNModel(this.isClassification, kVal, 'euclidean', 'distance', subSeed);
                    subModel.fit(trainX, trainY);
                } else if (mId === 'nb' && this.isClassification) {
                    subModel = new GaussianNaiveBayes(1e-4, 'empirical', subSeed);
                    subModel.fit(trainX, trainY);
                }

                if (subModel) {
                    if (subModel.usedFallback) {
                        this.fallbackNote = `Ensemble base neural network fell back to a linear model (${subModel.fallbackReason}).`;
                    }
                    models.push({ id: mId, model: subModel });
                }
                // Yield to the browser between base models so the UI (and Force Stop) stays responsive
                await new Promise(r => setTimeout(r, 0));
            }
            return models;
        }

        /**
         * testX / testY are accepted for API compatibility but are deliberately NOT used:
         * performance weights are estimated on an internal validation split of the TRAINING data,
         * so the test set never influences the model.
         */
        async fit(trainX, trainY, testX = null, testY = null, onProgress = null) {
            this.models = [];
            this.weights = [];
            this.baseMetrics = [];
            this.weightingNote = null;
            this.fallbackNote = null;
            this.rng = createPRNG(this.seed);

            const power = this.userHyperparams.power || 2;
            let rawWeights = null;
            const n = trainX.length;

            if (this.voting === 'weighted' && n >= 10) {
                // Internal 80/20 validation split of the training data (stratified by class when classifying)
                const idx = Array.from({ length: n }, (_, i) => i);
                for (let i = n - 1; i > 0; i--) {
                    const j = Math.floor(this.rng() * (i + 1));
                    [idx[i], idx[j]] = [idx[j], idx[i]];
                }
                const innerTrain = [];
                const innerVal = [];
                if (this.isClassification) {
                    const byClass = {};
                    idx.forEach(i => { (byClass[trainY[i]] = byClass[trainY[i]] || []).push(i); });
                    Object.values(byClass).forEach(list => {
                        const cut = list.length > 1 ? Math.max(1, Math.round(list.length * 0.8)) : list.length;
                        innerTrain.push(...list.slice(0, cut));
                        innerVal.push(...list.slice(cut));
                    });
                } else {
                    const cut = Math.round(n * 0.8);
                    innerTrain.push(...idx.slice(0, cut));
                    innerVal.push(...idx.slice(cut));
                }

                if (innerVal.length > 0 && innerTrain.length > 1) {
                    const itX = innerTrain.map(i => trainX[i]);
                    const itY = innerTrain.map(i => trainY[i]);
                    const ivX = innerVal.map(i => trainX[i]);
                    const ivY = innerVal.map(i => trainY[i]);
                    const weightingModels = await this._trainBaseModels(itX, itY, onProgress, 'Weighting phase · ');
                    rawWeights = [];
                    weightingModels.forEach(item => {
                        const preds = item.model.predict(ivX);
                        if (this.isClassification) {
                            // Balanced accuracy (mean per-class recall) so a model that ignores a
                            // clinically important minority class is not rewarded
                            const perClass = {};
                            for (let i = 0; i < ivY.length; i++) {
                                const c = ivY[i];
                                if (!perClass[c]) perClass[c] = { n: 0, hit: 0 };
                                perClass[c].n++;
                                if (preds[i] === c) perClass[c].hit++;
                            }
                            const recalls = Object.values(perClass).map(o => o.hit / o.n);
                            const balAcc = recalls.reduce((a, b) => a + b, 0) / (recalls.length || 1);
                            const w = Math.pow(Math.max(0.05, balAcc), power);
                            rawWeights.push(w);
                            this.baseMetrics.push({ id: item.id, balancedAccuracy: balAcc, weight: w });
                        } else {
                            let sumSq = 0;
                            for (let i = 0; i < ivY.length; i++) sumSq += Math.pow(ivY[i] - preds[i], 2);
                            const mse = sumSq / ivY.length;
                            const w = Math.pow(1.0 / (mse + 0.001), power / 2);
                            rawWeights.push(w);
                            this.baseMetrics.push({ id: item.id, mse, weight: w });
                        }
                    });
                    this.weightingNote = `Weights estimated on an internal ${innerVal.length}-case validation split of the training data.`;
                    // Weighting-phase models are no longer needed: free their GPU memory
                    weightingModels.forEach(item => {
                        if (item.model && typeof item.model.dispose === 'function') item.model.dispose();
                    });
                }
            }

            // Final base models are always fitted on the full training data
            this.models = await this._trainBaseModels(trainX, trainY, onProgress, rawWeights ? 'Final fit · ' : '');

            if (!rawWeights || rawWeights.length !== this.models.length) {
                if (this.voting === 'weighted') {
                    this.weightingNote = 'Too few training cases for an internal validation split; equal weights used.';
                }
                this.baseMetrics = [];
                rawWeights = new Array(this.models.length).fill(1.0);
                this.models.forEach(item => {
                    this.baseMetrics.push({ id: item.id, weight: 1.0 });
                });
            }

            const totalW = rawWeights.reduce((a, b) => a + b, 0) || 1.0;
            this.weights = rawWeights.map(w => w / totalW);
            this.baseMetrics.forEach((m, i) => {
                m.normalizedWeight = Number(this.weights[i].toFixed(3));
            });
        }

        predictSingle(x) {
            if (!this.models || this.models.length === 0) {
                throw new Error("Ensemble model is not trained yet.");
            }
            return this._combine(this.models.map(item => item.model.predictSingle(x)));
        }

        // Batched prediction: each base model predicts all rows at once (one GPU pass for the neural network)
        predictOutputs(X) {
            if (!this.models || this.models.length === 0) {
                throw new Error("Ensemble model is not trained yet.");
            }
            const perModel = this.models.map(item => batchOutputs(item.model, X));
            return X.map((_, r) => this._combine(perModel.map(outs => outs[r])));
        }

        // Combines one row's base-model outputs (baseResults[i] belongs to this.models[i])
        _combine(baseResults) {
            if (this.isClassification) {
                if (this.voting === 'hard') {
                    const votes = new Array(this.numClasses).fill(0);
                    this.models.forEach((item, i) => {
                        const res = baseResults[i];
                        const predClass = res.pred;
                        if (predClass >= 0 && predClass < this.numClasses) {
                            votes[predClass] += this.weights[i];
                        }
                    });

                    let bestClass = 0;
                    let maxVotes = -1;
                    votes.forEach((v, c) => {
                        if (v > maxVotes) {
                            maxVotes = v;
                            bestClass = c;
                        }
                    });

                    const totalVotes = votes.reduce((a, b) => a + b, 0) || 1;
                    const probMap = {};
                    votes.forEach((v, c) => {
                        probMap[c] = Number((v / totalVotes).toFixed(4));
                    });

                    return {
                        pred: bestClass,
                        probabilities: probMap,
                        confidence: Number((maxVotes / totalVotes).toFixed(4)),
                        probabilityType: 'vote',
                        ensembleDetails: this.baseMetrics
                    };
                } else {
                    // Soft voting (weighted probability average)
                    const combinedProbs = new Array(this.numClasses).fill(0);

                    this.models.forEach((item, i) => {
                        const res = baseResults[i];
                        const w = this.weights[i];

                        if (res.probabilities) {
                            for (let c = 0; c < this.numClasses; c++) {
                                const p = res.probabilities[c] !== undefined ? res.probabilities[c] : (res.pred === c ? 1 : 0);
                                combinedProbs[c] += w * p;
                            }
                        } else {
                            combinedProbs[res.pred] += w;
                        }
                    });

                    let bestClass = 0;
                    let maxP = -1;
                    const probMap = {};

                    combinedProbs.forEach((p, c) => {
                        const clampedP = Number(Math.max(0, Math.min(1, p)).toFixed(4));
                        probMap[c] = clampedP;
                        if (clampedP > maxP) {
                            maxP = clampedP;
                            bestClass = c;
                        }
                    });

                    return {
                        pred: bestClass,
                        probabilities: probMap,
                        confidence: Number(maxP.toFixed(4)),
                        probabilityType: 'blend',
                        ensembleDetails: this.baseMetrics
                    };
                }
            } else {
                // Regression: weighted average of continuous outputs
                let weightedSum = 0;
                const subPredictions = [];

                this.models.forEach((item, i) => {
                    const res = baseResults[i];
                    weightedSum += this.weights[i] * res.pred;
                    subPredictions.push({ id: item.id, pred: res.pred, weight: this.weights[i] });
                });

                return {
                    pred: Number(weightedSum.toFixed(4)),
                    confidence: null,
                    subPredictions,
                    ensembleDetails: this.baseMetrics
                };
            }
        }

        predict(X) {
            return this.predictOutputs(X).map(o => o.pred);
        }

        // Frees GPU memory held by neural-network base models
        dispose() {
            (this.models || []).forEach(item => {
                if (item.model && typeof item.model.dispose === 'function') item.model.dispose();
            });
            this.models = [];
        }
    }

    // ==========================================
    // 7. MODEL MANAGER & EVALUATION METRICS
    // ==========================================
    const ModelEngine = {
        SUPPORTED_MODELS: [
            { id: 'ensemble', name: 'Voting / Performance-Weighted Ensemble', type: 'both', desc: 'Combines the selected base models (neural network, random forest, logistic/ridge, k-NN, naive Bayes) by soft probability averaging, hard majority voting, or weights learned on an internal validation split of the training data. Not a cross-validated stacked Super Learner.' },
            { id: 'nn', name: 'Neural Network (Multilayer Perceptron, TF.js)', type: 'both', desc: 'Multilayer perceptron with dense hidden layers, dropout, L2 weight decay and early stopping on an internal validation split. Falls back to a linear model (and says so) if TensorFlow.js fails.' },
            { id: 'rf', name: 'Random Forest (Ensemble Trees)', type: 'both', desc: 'Bootstrap-sampled decision trees with random feature subsets. Splits use Gini impurity (classification) or variance reduction (regression) over an approximate grid of about 10 candidate thresholds per feature. Feature importance is impurity-based (MDI): useful for ranking, not evidence of causal or clinical effect.' },
            { id: 'glm', name: 'Logistic / Ridge Regression', type: 'both', desc: 'L2-regularized linear (ridge) or logistic/softmax model fitted by gradient descent on standardised features.' },
            { id: 'knn', name: 'k-Nearest Neighbors (k-NN)', type: 'both', desc: 'Non-parametric distance-based model; "probabilities" are (weighted) neighbour vote shares.' },
            { id: 'nb', name: 'Naive Bayes (Gaussian + Bernoulli)', type: 'classification', desc: 'Naive Bayes with Gaussian likelihoods for continuous features and Laplace-smoothed Bernoulli likelihoods for 0/1 indicator features (one-hot categories, binary flags).' }
        ],

        /**
         * Create an instance of a model based on user selection
         */
        createModel(modelId, isClassification, numClasses = 2, userHyperparams = {}) {
            let model;
            const seed = userHyperparams.seed !== undefined && userHyperparams.seed !== null ? userHyperparams.seed : null;
            if (modelId === 'ensemble') {
                model = new EnsembleModel(isClassification, numClasses, userHyperparams);
            } else if (modelId === 'nn') {
                model = new DeepNeuralNetwork(isClassification, numClasses, userHyperparams);
            } else if (modelId === 'rf') {
                const trees = userHyperparams.numTrees || 30;
                const depth = userHyperparams.maxDepth || 6;
                const minSplit = userHyperparams.minSamplesSplit || 2;
                const featRatio = userHyperparams.featureRatio || 'sqrt';
                model = new RandomForest(isClassification, trees, depth, minSplit, featRatio, seed);
            } else if (modelId === 'glm') {
                model = new GeneralizedLinearModel(
                    isClassification,
                    numClasses,
                    userHyperparams.lr || 0.05,
                    userHyperparams.epochs || 150,
                    userHyperparams.lambdaL2 !== undefined ? userHyperparams.lambdaL2 : 0.01,
                    userHyperparams.solver || 'momentum',
                    seed
                );
            } else if (modelId === 'knn') {
                model = new KNNModel(
                    isClassification,
                    userHyperparams.k || 5,
                    userHyperparams.metric || 'euclidean',
                    userHyperparams.weighting || 'uniform',
                    seed
                );
            } else if (modelId === 'nb') {
                if (!isClassification) throw new Error("Naive Bayes is only supported for classification tasks.");
                model = new GaussianNaiveBayes(
                    userHyperparams.varSmoothing !== undefined ? userHyperparams.varSmoothing : 1e-4,
                    userHyperparams.priorMode || 'empirical',
                    seed
                );
            } else {
                throw new Error(`Unknown model ID: ${modelId}`);
            }
            if (model && seed !== null && seed !== undefined) {
                model.seed = seed;
            }
            return model;
        },

        /**
         * Evaluate model performance on a held-out set.
         * options.regressionThreshold: cutoff for the exploratory 2×2 table of a continuous outcome.
         *   It should come from the TRAINING data (or be a prespecified clinical cutoff); if omitted,
         *   no 2×2 table is produced.
         * options.thresholdSource: free-text description of where that cutoff came from.
         */
        evaluate(model, testX, testY, isClassification, targetClasses = [], options = {}) {
            if (!testX || testX.length === 0) {
                return null;
            }

            if (isClassification) {
                const numClasses = targetClasses.length || [...new Set(testY)].length;
                const total = testX.length;
                const outputs = batchOutputs(model, testX);
                const preds = outputs.map(o => o.pred);
                let correct = 0;

                // Confusion Matrix [actual][pred]
                const confMatrix = Array.from({ length: numClasses }, () => new Array(numClasses).fill(0));

                for (let i = 0; i < total; i++) {
                    const actual = testY[i];
                    const pred = preds[i];
                    if (actual === pred) correct++;
                    if (actual < numClasses && pred < numClasses) {
                        confMatrix[actual][pred]++;
                    }
                }

                const accuracyRaw = correct / total;
                const accuracy = Number(accuracyRaw.toFixed(4));

                // Wilson score 95% confidence interval for accuracy
                const z = 1.959964;
                const denom = 1 + (z * z) / total;
                const centre = (accuracyRaw + (z * z) / (2 * total)) / denom;
                const half = (z * Math.sqrt((accuracyRaw * (1 - accuracyRaw)) / total + (z * z) / (4 * total * total))) / denom;
                const accuracyCI = [Number(Math.max(0, centre - half).toFixed(4)), Number(Math.min(1, centre + half).toFixed(4))];

                // Per-class metrics
                const classList = targetClasses.length ? targetClasses : Array.from({ length: numClasses }, (_, i) => String(i));
                const classMetrics = classList.map((clsName, cIdx) => {
                    let tp = confMatrix[cIdx][cIdx];
                    let fp = 0;
                    let fn = 0;
                    let support = 0;

                    for (let r = 0; r < numClasses; r++) {
                        support += confMatrix[cIdx][r];
                        if (r !== cIdx) {
                            fn += confMatrix[cIdx][r];
                            fp += confMatrix[r][cIdx];
                        }
                    }

                    const precision = (tp + fp) > 0 ? Number((tp / (tp + fp)).toFixed(4)) : 0;
                    const recall = (tp + fn) > 0 ? Number((tp / (tp + fn)).toFixed(4)) : 0;
                    const f1 = (precision + recall) > 0 ? Number(((2 * precision * recall) / (precision + recall)).toFixed(4)) : 0;

                    return {
                        className: clsName,
                        precision,
                        recall,
                        f1,
                        support
                    };
                });

                // Macro-averaged metrics (all classes, as in scikit-learn's default)
                const macroPrecision = Number((classMetrics.reduce((a, b) => a + b.precision, 0) / numClasses).toFixed(4));
                const macroRecall = Number((classMetrics.reduce((a, b) => a + b.recall, 0) / numClasses).toFixed(4));
                const macroF1 = Number((classMetrics.reduce((a, b) => a + b.f1, 0) / numClasses).toFixed(4));
                // Balanced accuracy: mean recall over classes present in the test set
                const present = classMetrics.filter(c => c.support > 0);
                const balancedAccuracy = present.length ? Number((present.reduce((a, b) => a + b.recall, 0) / present.length).toFixed(4)) : null;
                const classesMissingFromTest = classMetrics.filter(c => c.support === 0).map(c => c.className);

                // Class-probability vectors (vote shares for RF / hard voting; neighbour weights for k-NN)
                const probVectors = outputs.map(o => {
                    const v = new Array(numClasses).fill(0);
                    if (o.probabilities) {
                        for (let c = 0; c < numClasses; c++) {
                            const p = Number(o.probabilities[c]);
                            v[c] = isNaN(p) ? 0 : p;
                        }
                    } else if (o.pred >= 0 && o.pred < numClasses) {
                        v[o.pred] = 1;
                    }
                    return v;
                });

                // Brier score (lower is better): binary = mean (p_positive − y)²; multiclass = mean Σ_c (p_c − y_c)²
                let brierScore = null;
                if (numClasses === 2) {
                    brierScore = probVectors.reduce((acc, v, i) => acc + Math.pow(v[1] - (testY[i] === 1 ? 1 : 0), 2), 0) / total;
                } else if (numClasses > 2) {
                    brierScore = probVectors.reduce((acc, v, i) => acc + v.reduce((s2, p, c) => s2 + Math.pow(p - (testY[i] === c ? 1 : 0), 2), 0), 0) / total;
                }
                if (brierScore !== null) brierScore = Number(brierScore.toFixed(4));

                // ROC-AUC for binary classification
                let rocAuc = null;
                if (numClasses === 2) {
                    try {
                        rocAuc = this._calculateBinaryAUC(testY, probVectors.map(v => v[1]));
                    } catch (e) {
                        rocAuc = null;
                    }
                }

                const probabilityType = outputs.length ? (outputs[0].probabilityType || 'model') : 'model';

                return {
                    isClassification: true,
                    totalSamples: total,
                    accuracy,
                    accuracyCI,
                    balancedAccuracy,
                    macroPrecision,
                    macroRecall,
                    macroF1,
                    rocAuc,
                    brierScore,
                    probabilityType,
                    classesMissingFromTest,
                    // For binary outcomes, class index 1 is the "positive" class for AUC / sensitivity / TP
                    positiveClass: numClasses === 2 && targetClasses.length === 2 ? targetClasses[1] : null,
                    confMatrix,
                    classMetrics,
                    actuals: testY,
                    predictions: preds,
                    seed: model.seed !== undefined ? model.seed : null
                };
            } else {
                // Regression Metrics: R2, MSE, RMSE, MAE
                const preds = model.predict(testX);
                const total = testX.length;
                let sumSqErr = 0;
                let sumAbsErr = 0;
                const meanActual = testY.reduce((a, b) => a + b, 0) / total;
                let totalSq = 0;

                for (let i = 0; i < total; i++) {
                    const actual = testY[i];
                    const pred = preds[i];
                    const err = actual - pred;
                    sumSqErr += err * err;
                    sumAbsErr += Math.abs(err);
                    totalSq += Math.pow(actual - meanActual, 2);
                }

                const mse = Number((sumSqErr / total).toFixed(4));
                const rmse = Number((Math.sqrt(sumSqErr / total)).toFixed(4));
                const mae = Number((sumAbsErr / total).toFixed(4));
                // R² is undefined when the test outcomes have no variance
                const r2 = totalSq > 0 ? Number((1 - (sumSqErr / totalSq)).toFixed(4)) : null;

                const result = {
                    isClassification: false,
                    totalSamples: total,
                    r2,
                    mse,
                    rmse,
                    mae,
                    actuals: testY,
                    predictions: preds,
                    seed: model.seed !== undefined ? model.seed : null
                };

                // Exploratory 2×2 dichotomisation at a cutoff fixed BEFORE looking at the test outcomes
                const threshold = options.regressionThreshold;
                if (typeof threshold === 'number' && isFinite(threshold)) {
                    // Row 0 = Actual < cutoff, Row 1 = Actual >= cutoff; Col 0/1 likewise for predictions
                    const confMatrix2x2 = [[0, 0], [0, 0]];
                    let hits = 0;
                    for (let i = 0; i < total; i++) {
                        const actBin = testY[i] >= threshold ? 1 : 0;
                        const predBin = preds[i] >= threshold ? 1 : 0;
                        confMatrix2x2[actBin][predBin]++;
                        if (actBin === predBin) hits++;
                    }
                    const tp = confMatrix2x2[1][1];
                    const fn = confMatrix2x2[1][0];
                    const fp = confMatrix2x2[0][1];
                    const tn = confMatrix2x2[0][0];
                    result.confMatrix = confMatrix2x2;
                    result.threshold = Number(threshold.toFixed(4));
                    result.thresholdSource = options.thresholdSource || 'user-supplied cutoff';
                    result.thresholdAccuracy = Number((hits / total).toFixed(4));
                    // Undefined (null) when no test cases fall on that side of the cutoff
                    result.sensitivity = (tp + fn) > 0 ? Number((tp / (tp + fn)).toFixed(4)) : null;
                    result.specificity = (tn + fp) > 0 ? Number((tn / (tn + fp)).toFixed(4)) : null;
                }
                return result;
            }
        },

        predictOutputs(model, X) {
            return batchOutputs(model, X);
        },

        evaluateModel(model, testX, testY, isClassificationOrClasses = true, targetClassesOrStats = []) {
            let isClassification = true;
            let targetClasses = [];
            if (typeof isClassificationOrClasses === 'boolean') {
                isClassification = isClassificationOrClasses;
                targetClasses = Array.isArray(targetClassesOrStats) ? targetClassesOrStats : [];
            } else if (Array.isArray(isClassificationOrClasses)) {
                targetClasses = isClassificationOrClasses;
                isClassification = targetClasses.length > 0;
            } else if (model && model.isClassification !== undefined) {
                isClassification = Boolean(model.isClassification);
                targetClasses = Array.isArray(targetClassesOrStats) ? targetClassesOrStats : [];
            }
            return this.evaluate(model, testX, testY, isClassification, targetClasses);
        },

        _calculateBinaryAUC(yTrue, yScores) {
            // Rank-based Mann-Whitney U test calculation for ROC AUC
            const items = yTrue.map((val, idx) => ({ y: val, score: yScores[idx] }));
            items.sort((a, b) => a.score - b.score);

            let nPos = 0;
            let nNeg = 0;
            items.forEach(it => {
                if (it.y === 1) nPos++;
                else nNeg++;
            });

            // AUC is undefined when the test set contains only one class
            if (nPos === 0 || nNeg === 0) return null;

            // Mid-ranks for tied scores (ties are common with vote-based probabilities)
            let rankSumPos = 0;
            let i = 0;
            while (i < items.length) {
                let j = i;
                while (j + 1 < items.length && items[j + 1].score === items[i].score) j++;
                const avgRank = (i + j + 2) / 2;
                for (let k = i; k <= j; k++) {
                    if (items[k].y === 1) rankSumPos += avgRank;
                }
                i = j + 1;
            }

            const u = rankSumPos - (nPos * (nPos + 1)) / 2;
            const auc = u / (nPos * nNeg);
            return Number(Math.max(0, Math.min(1, auc)).toFixed(4));
        }
    };

    window.ModelEngine = ModelEngine;
    window.ModelFactory = ModelEngine;
    window.RandomForest = RandomForest;
    window.KNNModel = KNNModel;
    window.GeneralizedLinearModel = GeneralizedLinearModel;
    window.DeepNeuralNetwork = DeepNeuralNetwork;
    window.GaussianNaiveBayes = GaussianNaiveBayes;
    window.EnsembleModel = EnsembleModel;
})(window);
