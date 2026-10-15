/**
 * NPH ML/Neuralnet Predictor - Data Engine
 * Handles CSV parsing, variable role & type deduction, missing value detection,
 * multi-strategy data imputation, feature encoding, scaling, and train/test splitting.
 * 
 * Conceived, designed and developed by Dr G Narenthiran MB ChB BSc(MedSci)(Hons) FEBNS FRCS(SN)
 */

(function (window) {
    'use strict';

    const TARGET_NAME_RE = /^(target|outcome|response|shunt_response|label|class|diagnos|result|improvement)/i;

    // Conventional clinical ordering words (lowest rank first) used to order text ordinal levels
    const ORDINAL_WORD_ORDER = [
        'none', 'absent', 'never', 'normal', 'very low', 'low', 'minimal', 'mild', 'slight',
        'moderate', 'medium', 'intermediate', 'average', 'high', 'marked', 'severe', 'very high', 'very severe', 'critical', 'extreme'
    ];

    function ordinalWordRank(value) {
        const v = String(value).trim().toLowerCase();
        let best = -1;
        let bestLen = 0;
        ORDINAL_WORD_ORDER.forEach((w, idx) => {
            // Longest matching word wins (so "very high" beats "high")
            if ((v === w || v.startsWith(w + ' ') || v.endsWith(' ' + w) || v.includes(' ' + w + ' ')) && w.length > bestLen) {
                best = idx;
                bestLen = w.length;
            }
        });
        return best;
    }

    function arrayMin(arr) {
        let m = Infinity;
        for (let i = 0; i < arr.length; i++) if (arr[i] < m) m = arr[i];
        return m;
    }

    function arrayMax(arr) {
        let m = -Infinity;
        for (let i = 0; i < arr.length; i++) if (arr[i] > m) m = arr[i];
        return m;
    }

    const DataEngine = {
        /**
         * Orders ordinal levels: numerically when all numeric; by conventional clinical words
         * (none < mild < moderate < severe, low < medium < high) when every level is recognised;
         * otherwise by natural (numeric-aware) text order, so "Grade 2" < "Grade 10".
         */
        sortOrdinalLevels(values) {
            const levels = Array.from(new Set(values.map(v => (typeof v === 'number' ? v : String(v)))));
            if (levels.every(v => !isNaN(Number(v)) && String(v).trim() !== '')) {
                return levels.sort((a, b) => Number(a) - Number(b));
            }
            const wordRanks = levels.map(ordinalWordRank);
            const rankSet = new Set(wordRanks);
            if (wordRanks.every(r => r >= 0) && rankSet.size === levels.length) {
                return levels
                    .map((v, i) => ({ v, r: wordRanks[i] }))
                    .sort((a, b) => a.r - b.r)
                    .map(o => o.v);
            }
            return levels.sort((a, b) => String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: 'base' }));
        },

        /**
         * Builds a value -> rank lookup for an ordinal column
         */
        buildOrdinalRankMap(values) {
            const ranks = this.sortOrdinalLevels(values);
            const rankMap = {};
            ranks.forEach((val, rk) => { rankMap[String(val)] = rk; });
            return { ranks, rankMap };
        },

        /**
         * RFC 4180 compliant CSV Parser with auto delimiter detection
         */
        parseCSV(csvText) {
            if (!csvText || typeof csvText !== 'string') {
                throw new Error("Empty or invalid CSV file content.");
            }

            // Detect delimiter: check first 3 lines
            const sampleLines = csvText.split(/\r\n|\n|\r/).filter(l => l.trim().length > 0).slice(0, 3);
            let delimiter = ',';
            if (sampleLines.length > 0) {
                const commaCount = (sampleLines[0].match(/,/g) || []).length;
                const semiCount = (sampleLines[0].match(/;/g) || []).length;
                const tabCount = (sampleLines[0].match(/\t/g) || []).length;
                if (semiCount > commaCount && semiCount >= tabCount) delimiter = ';';
                else if (tabCount > commaCount && tabCount >= semiCount) delimiter = '\t';
            }

            const rows = [];
            let currentRow = [];
            let currentField = '';
            let inQuotes = false;
            let i = 0;
            const len = csvText.length;

            while (i < len) {
                const char = csvText[i];
                const nextChar = csvText[i + 1];

                if (inQuotes) {
                    if (char === '"') {
                        if (nextChar === '"') {
                            currentField += '"';
                            i += 2;
                            continue;
                        } else {
                            inQuotes = false;
                            i++;
                            continue;
                        }
                    } else {
                        currentField += char;
                        i++;
                        continue;
                    }
                } else {
                    if (char === '"') {
                        inQuotes = true;
                        i++;
                        continue;
                    } else if (char === delimiter) {
                        currentRow.push(currentField.trim());
                        currentField = '';
                        i++;
                        continue;
                    } else if (char === '\r' || char === '\n') {
                        currentRow.push(currentField.trim());
                        currentField = '';
                        if (char === '\r' && nextChar === '\n') {
                            i += 2;
                        } else {
                            i++;
                        }
                        if (currentRow.some(val => val.length > 0)) {
                            rows.push(currentRow);
                        }
                        currentRow = [];
                        continue;
                    } else {
                        currentField += char;
                        i++;
                        continue;
                    }
                }
            }

            if (currentField.length > 0 || currentRow.length > 0) {
                currentRow.push(currentField.trim());
                if (currentRow.some(val => val.length > 0)) {
                    rows.push(currentRow);
                }
            }

            if (rows.length < 2) {
                throw new Error("CSV must contain at least a header row and one data row.");
            }

            const rawHeaders = rows[0];
            // Deduplicate headers if any
            const seenHeaders = {};
            const headers = rawHeaders.map((h, idx) => {
                let name = (h || `Column_${idx + 1}`).trim();
                if (seenHeaders[name]) {
                    seenHeaders[name]++;
                    name = `${name}_${seenHeaders[name]}`;
                } else {
                    seenHeaders[name] = 1;
                }
                return name;
            });

            const dataRows = [];
            for (let r = 1; r < rows.length; r++) {
                const row = rows[r];
                const obj = {};
                for (let c = 0; c < headers.length; c++) {
                    const rawVal = c < row.length ? row[c] : '';
                    obj[headers[c]] = rawVal;
                }
                dataRows.push(obj);
            }

            return {
                headers,
                rows: dataRows,
                delimiter,
                totalRows: dataRows.length,
                totalCols: headers.length
            };
        },

        /**
         * Checks if a cell value is missing / blank / NaN / NA
         */
        isMissing(val) {
            if (val === null || val === undefined) return true;
            const str = String(val).trim();
            if (str === '') return true;
            const lower = str.toLowerCase();
            return lower === 'na' || lower === 'nan' || lower === 'null' || lower === 'none' || lower === '?' || lower === '.' || lower === 'n/a';
        },

        /**
         * Analyzes dataset columns: missing counts, data types, distinct values, and heuristics
         */
        analyzeColumns(headers, rows) {
            const analysis = {};

            headers.forEach((col, colIdx) => {
                let missingCount = 0;
                const distinctMap = new Map();
                let numericCount = 0;
                let validCount = 0;
                const validNumbers = [];

                rows.forEach((row, rowIdx) => {
                    const val = row[col];
                    if (this.isMissing(val)) {
                        missingCount++;
                    } else {
                        validCount++;
                        const strVal = String(val).trim();
                        distinctMap.set(strVal, (distinctMap.get(strVal) || 0) + 1);

                        const num = Number(strVal);
                        if (!isNaN(num) && strVal !== '') {
                            numericCount++;
                            validNumbers.push(num);
                        }
                    }
                });

                const isNumeric = validCount > 0 && numericCount === validCount;
                const distinctCount = distinctMap.size;
                const distinctValues = Array.from(distinctMap.keys());

                // Heuristic Role determination
                let role = 'feature';
                const lowerCol = col.toLowerCase();

                if (lowerCol.match(/^(patient_id|id|mrn|record_id|patient|subjid|subject|hash)$/i) || (colIdx === 0 && distinctCount === validCount && !isNumeric)) {
                    role = 'meta';
                } else if (colIdx === headers.length - 1 || TARGET_NAME_RE.test(lowerCol)) {
                    role = 'target';
                }

                // Heuristic Type determination
                let type = 'nominal';
                if (isNumeric) {
                    const allIntegers = validNumbers.every(n => Number.isInteger(n));
                    if (distinctCount <= 7 && allIntegers && distinctCount >= 2) {
                        type = 'ordinal';
                    } else if (distinctCount > 7 || !allIntegers) {
                        type = 'continuous';
                    } else {
                        type = 'nominal';
                    }
                } else {
                    // Check if string labels resemble ordinal grades (e.g. grade 1, grade 2 or mild, moderate, severe)
                    const lowerValues = distinctValues.map(v => v.toLowerCase());
                    const looksOrdinal = lowerValues.some(v => v.includes('mild') || v.includes('mod') || v.includes('sev') || v.includes('stage') || v.includes('grade') || v.match(/^[0-9]+$/));
                    if (looksOrdinal && distinctCount <= 7) {
                        type = 'ordinal';
                    } else {
                        type = 'nominal';
                    }
                }

                // Calculate summary stats for numbers
                let min = null, max = null, mean = null, median = null, mode = null;
                if (validNumbers.length > 0) {
                    validNumbers.sort((a, b) => a - b);
                    min = validNumbers[0];
                    max = validNumbers[validNumbers.length - 1];
                    const sum = validNumbers.reduce((acc, v) => acc + v, 0);
                    mean = Number((sum / validNumbers.length).toFixed(4));
                    const mid = Math.floor(validNumbers.length / 2);
                    median = validNumbers.length % 2 === 0 ? Number(((validNumbers[mid - 1] + validNumbers[mid]) / 2).toFixed(4)) : validNumbers[mid];
                }

                // Calculate mode for any type
                let bestFreq = 0;
                distinctMap.forEach((freq, val) => {
                    if (freq > bestFreq) {
                        bestFreq = freq;
                        mode = isNumeric ? Number(val) : val;
                    }
                });

                analysis[col] = {
                    name: col,
                    index: colIdx,
                    total: rows.length,
                    validCount,
                    missingCount,
                    missingPercent: Number(((missingCount / rows.length) * 100).toFixed(1)),
                    distinctCount,
                    distinctValues,
                    isNumeric,
                    role, // 'meta', 'target', 'feature'
                    type, // 'continuous', 'ordinal', 'nominal'
                    stats: {
                        min,
                        max,
                        mean,
                        median,
                        mode,
                        distribution: Object.fromEntries(distinctMap)
                    },
                    // Default imputation strategy suggestion
                    suggestedImputation: type === 'continuous' ? 'median' : 'mode'
                };
            });

            // Only one target may be suggested: prefer an outcome-like column name, else the last column
            const targetCols = headers.filter(h => analysis[h].role === 'target');
            if (targetCols.length > 1) {
                const keep = targetCols.find(h => TARGET_NAME_RE.test(h.toLowerCase())) || targetCols[targetCols.length - 1];
                targetCols.forEach(h => {
                    if (h !== keep) analysis[h].role = 'feature';
                });
            }

            return analysis;
        },

        /**
         * Perform data imputation across columns based on user configuration
         * @param {Array} rows - array of row objects
         * @param {Object} columnConfigs - { [colName]: { role, type, imputationStrategy, customImputeVal } }
         * @param {Object} columnAnalysis - analysis output from analyzeColumns
         */
        imputeDataset(rows, columnConfigs, columnAnalysis) {
            let workingRows = rows.map(r => ({ ...r }));
            const audit = [];
            const rowsToDrop = new Set();
            const targetMissingRows = new Set();

            // First pass: identify rows that must be dropped.
            // Rows with a missing outcome (target) are always excluded: an outcome must never be imputed.
            Object.keys(columnConfigs).forEach(col => {
                const config = columnConfigs[col];
                if (config.role === 'meta') return; // skip meta columns

                if (config.role === 'target') {
                    workingRows.forEach((row, rIdx) => {
                        if (this.isMissing(row[col])) targetMissingRows.add(rIdx);
                    });
                } else if (config.imputationStrategy === 'drop') {
                    workingRows.forEach((row, rIdx) => {
                        if (this.isMissing(row[col])) {
                            rowsToDrop.add(rIdx);
                        }
                    });
                }
            });

            if (targetMissingRows.size > 0) {
                audit.push({
                    type: 'drop_rows',
                    description: `Excluded ${targetMissingRows.size} row(s) with a missing outcome (target values are never imputed).`
                });
            }
            const featureOnlyDrops = [...rowsToDrop].filter(idx => !targetMissingRows.has(idx)).length;
            if (featureOnlyDrops > 0) {
                audit.push({
                    type: 'drop_rows',
                    description: `Dropped ${featureOnlyDrops} row(s) with missing values in columns marked for row deletion.`
                });
            }
            if (targetMissingRows.size > 0 || rowsToDrop.size > 0) {
                workingRows = workingRows.filter((_, idx) => !rowsToDrop.has(idx) && !targetMissingRows.has(idx));
            }

            // Second pass: Compute imputation values on the filtered rows
            Object.keys(columnConfigs).forEach(col => {
                const config = columnConfigs[col];
                if (config.role === 'meta') return;
                if (config.role === 'target') return;
                if (config.imputationStrategy === 'drop') return;

                let strategy = config.imputationStrategy;
                const colType = config.type;

                // Collect valid values from workingRows
                const validValues = [];
                const validNumbers = [];

                workingRows.forEach(r => {
                    const val = r[col];
                    if (!this.isMissing(val)) {
                        validValues.push(val);
                        const n = Number(val);
                        if (!isNaN(n)) validNumbers.push(n);
                    }
                });

                let fillValue = null;

                // Mean/median/KNN need numbers: text-valued columns fall back to the most frequent value
                if ((strategy === 'mean' || strategy === 'median' || strategy === 'knn') && validNumbers.length < validValues.length) {
                    strategy = 'mode';
                }

                if (strategy === 'mean') {
                    if (validNumbers.length > 0) {
                        const sum = validNumbers.reduce((a, b) => a + b, 0);
                        fillValue = Number((sum / validNumbers.length).toFixed(4));
                    } else {
                        fillValue = 0;
                    }
                } else if (strategy === 'median') {
                    if (validNumbers.length > 0) {
                        const sorted = [...validNumbers].sort((a, b) => a - b);
                        const mid = Math.floor(sorted.length / 2);
                        if (colType === 'ordinal') {
                            // Lower median keeps the fill value on an observed ordinal level
                            fillValue = sorted[Math.floor((sorted.length - 1) / 2)];
                        } else {
                            fillValue = sorted.length % 2 === 0 ? Number(((sorted[mid - 1] + sorted[mid]) / 2).toFixed(4)) : sorted[mid];
                        }
                    } else {
                        fillValue = 0;
                    }
                } else if (strategy === 'mode') {
                    const freqMap = new Map();
                    validValues.forEach(v => freqMap.set(String(v), (freqMap.get(String(v)) || 0) + 1));
                    let topCount = 0;
                    let topVal = null;
                    freqMap.forEach((count, v) => {
                        if (count > topCount) {
                            topCount = count;
                            topVal = v;
                        }
                    });
                    if (colType === 'continuous' || colType === 'ordinal') {
                        fillValue = topVal !== null && !isNaN(Number(topVal)) ? Number(topVal) : 0;
                    } else {
                        fillValue = topVal !== null ? topVal : 'Unknown';
                    }
                } else if (strategy === 'constant') {
                    if (colType === 'continuous' || colType === 'ordinal') {
                        fillValue = Number(config.customImputeVal !== undefined && config.customImputeVal !== '' ? config.customImputeVal : 0);
                    } else {
                        fillValue = String(config.customImputeVal || 'Missing');
                    }
                } else if (strategy === 'knn') {
                    // KNN Imputation for continuous variables
                    // Will fill cell-by-cell in third pass
                    fillValue = 'knn_dynamic';
                }

                // Perform the imputation
                let imputedCount = 0;
                if (strategy === 'knn') {
                    // Precompute continuous reference columns
                    const numericCols = Object.keys(columnConfigs).filter(c =>
                        columnConfigs[c].type === 'continuous' && columnConfigs[c].role === 'feature' && c !== col);
                    // Standardise reference columns so no single unit/scale dominates the distance
                    const refScale = {};
                    numericCols.forEach(nCol => {
                        const vals = workingRows.map(r => r[nCol]).filter(v => !this.isMissing(v)).map(Number).filter(n => !isNaN(n));
                        const m = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
                        const sd = vals.length ? Math.sqrt(vals.reduce((a, b) => a + (b - m) * (b - m), 0) / vals.length) : 0;
                        refScale[nCol] = sd > 0 ? sd : 1;
                    });
                    workingRows.forEach((row, rowIdx) => {
                        if (this.isMissing(row[col])) {
                            // Find k nearest rows that have non-missing col
                            const k = 3;
                            const distances = [];
                            workingRows.forEach((otherRow, otherIdx) => {
                                if (otherIdx === rowIdx || this.isMissing(otherRow[col]) || isNaN(Number(otherRow[col]))) return;
                                let distSq = 0;
                                let matchedFeatures = 0;
                                numericCols.forEach(nCol => {
                                    if (!this.isMissing(row[nCol]) && !this.isMissing(otherRow[nCol])) {
                                        const d = (Number(row[nCol]) - Number(otherRow[nCol])) / refScale[nCol];
                                        if (!isNaN(d)) {
                                            distSq += d * d;
                                            matchedFeatures++;
                                        }
                                    }
                                });
                                if (matchedFeatures > 0) {
                                    // Mean squared distance over shared features, so rows sharing fewer features are not favoured
                                    distances.push({ val: Number(otherRow[col]), dist: Math.sqrt(distSq / matchedFeatures) });
                                }
                            });

                            if (distances.length > 0) {
                                distances.sort((a, b) => a.dist - b.dist);
                                const neighbors = distances.slice(0, Math.min(k, distances.length));
                                const avg = neighbors.reduce((acc, item) => acc + item.val, 0) / neighbors.length;
                                row[col] = Number(avg.toFixed(4));
                                imputedCount++;
                            } else {
                                // fallback to mean
                                const fallback = validNumbers.length > 0 ? (validNumbers.reduce((a, b) => a + b, 0) / validNumbers.length) : 0;
                                row[col] = Number(fallback.toFixed(4));
                                imputedCount++;
                            }
                        }
                    });
                } else {
                    workingRows.forEach(row => {
                        if (this.isMissing(row[col])) {
                            row[col] = fillValue;
                            imputedCount++;
                        } else if (colType === 'continuous' || colType === 'ordinal') {
                            const n = Number(row[col]);
                            if (!isNaN(n)) row[col] = n;
                        }
                    });
                }

                if (imputedCount > 0) {
                    audit.push({
                        type: 'impute_column',
                        column: col,
                        strategy,
                        imputedCount,
                        filledValue: fillValue === 'knn_dynamic' ? 'KNN: mean of 3 nearest neighbours (standardised distance)' : fillValue
                    });
                }
            });

            return {
                imputedRows: workingRows,
                audit,
                totalRowsAfter: workingRows.length
            };
        },

        /**
         * Preprocess features and target with optional explicit featureCols and targetCol
         */
        prepareDataset(rows, columnConfigs, featureCols = null, targetCol = null) {
            const configs = {};
            Object.keys(columnConfigs).forEach(c => {
                configs[c] = { ...columnConfigs[c] };
            });

            if (targetCol) {
                Object.keys(configs).forEach(c => {
                    if (configs[c].role === 'target') configs[c].role = 'meta';
                });
                if (configs[targetCol]) configs[targetCol].role = 'target';
                else configs[targetCol] = { role: 'target', type: 'nominal' };
            }

            if (featureCols && Array.isArray(featureCols)) {
                Object.keys(configs).forEach(c => {
                    if (c !== targetCol) configs[c].role = 'meta';
                });
                featureCols.forEach(c => {
                    if (configs[c]) configs[c].role = 'feature';
                    else configs[c] = { role: 'feature', type: 'continuous' };
                });
            }

            return this.prepareFeaturesAndTarget(rows, configs);
        },

        /**
         * Preprocess features and target into numeric arrays for machine learning models
         */
        prepareFeaturesAndTarget(rows, columnConfigs) {
            const featureCols = [];
            let targetCol = null;

            Object.keys(columnConfigs).forEach(col => {
                const conf = columnConfigs[col];
                if (conf.role === 'feature') {
                    featureCols.push(col);
                } else if (conf.role === 'target') {
                    targetCol = col;
                }
            });

            if (!targetCol) {
                throw new Error("Please select exactly one Target variable to predict.");
            }
            if (featureCols.length === 0) {
                throw new Error("Please select at least one Feature (predictor) variable.");
            }

            const targetConf = columnConfigs[targetCol];
            const isClassification = targetConf.type === 'nominal' || targetConf.type === 'ordinal';

            // Rows without a usable outcome cannot be used for training (outcomes are never imputed)
            const droppedMissingTarget = rows.filter(r => this.isMissing(r[targetCol]) ||
                (!isClassification && isNaN(Number(r[targetCol])))).length;
            if (droppedMissingTarget > 0) {
                rows = rows.filter(r => !this.isMissing(r[targetCol]) &&
                    (isClassification || !isNaN(Number(r[targetCol]))));
            }
            if (rows.length === 0) {
                throw new Error(`No rows with a valid value for the target '${targetCol}'.`);
            }

            // 1. Build Target Encoder
            let targetClasses = [];
            let targetClassToIdx = {};
            let targetIdxToClass = {};
            let targetStats = null;

            if (isClassification) {
                // Collect distinct target classes
                const classSet = new Set();
                rows.forEach(r => {
                    const val = String(r[targetCol]);
                    classSet.add(val);
                });
                targetClasses = Array.from(classSet);
                // Sort classes naturally
                targetClasses.sort();
                targetClasses.forEach((cls, idx) => {
                    targetClassToIdx[cls] = idx;
                    targetIdxToClass[idx] = cls;
                });
            } else {
                // Regression target: collect numbers
                const nums = rows.map(r => Number(r[targetCol])).filter(n => !isNaN(n));
                const sum = nums.reduce((a, b) => a + b, 0);
                const mean = sum / nums.length;
                const variance = nums.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / nums.length;
                targetStats = {
                    mean,
                    std: Math.sqrt(variance) || 1,
                    min: arrayMin(nums),
                    max: arrayMax(nums)
                };
            }

            // 2. Build Feature Encoders
            // For Nominal: One-Hot Encoding map
            // For Ordinal: Natural / sorted rank mapping
            // For Continuous: Standardization (mean, std)
            const featureMetadata = {};
            let totalEncodedFeatures = 0;
            const encodedFeatureNames = [];

            featureCols.forEach(col => {
                const conf = columnConfigs[col];
                if (conf.type === 'nominal') {
                    const catSet = new Set();
                    const catFreq = {};
                    rows.forEach(r => {
                        const c = String(r[col]);
                        catSet.add(c);
                        catFreq[c] = (catFreq[c] || 0) + 1;
                    });
                    const categories = Array.from(catSet).sort();
                    let modeCat = categories[0];
                    categories.forEach(c => { if (catFreq[c] > catFreq[modeCat]) modeCat = c; });
                    featureMetadata[col] = {
                        type: 'nominal',
                        categories,
                        mode: modeCat,
                        startIndex: totalEncodedFeatures,
                        length: categories.length
                    };
                    categories.forEach(cat => {
                        encodedFeatureNames.push(`${col}_${cat}`);
                    });
                    totalEncodedFeatures += categories.length;
                } else if (conf.type === 'ordinal') {
                    // Ordinal mapping: map ordered distinct levels to integers 0, 1, 2...
                    const { ranks, rankMap } = this.buildOrdinalRankMap(rows.map(r => r[col]));
                    // Median rank of the training rows: fallback for missing / unseen levels at prediction time
                    const rankVals = rows.map(r => rankMap[String(r[col])]).filter(v => v !== undefined).sort((a, b) => a - b);
                    const defaultRank = rankVals.length ? rankVals[Math.floor((rankVals.length - 1) / 2)] : 0;
                    featureMetadata[col] = {
                        type: 'ordinal',
                        ranks,
                        rankMap,
                        defaultRank,
                        startIndex: totalEncodedFeatures,
                        length: 1
                    };
                    encodedFeatureNames.push(col);
                    totalEncodedFeatures += 1;
                } else {
                    // Continuous: compute mean and standard deviation
                    const nums = rows.map(r => Number(r[col])).filter(n => !isNaN(n));
                    const sum = nums.reduce((a, b) => a + b, 0);
                    const mean = sum / (nums.length || 1);
                    const variance = nums.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (nums.length || 1);
                    const std = Math.sqrt(variance) || 1;
                    featureMetadata[col] = {
                        type: 'continuous',
                        mean,
                        std,
                        min: nums.length ? arrayMin(nums) : 0,
                        max: nums.length ? arrayMax(nums) : 0,
                        startIndex: totalEncodedFeatures,
                        length: 1
                    };
                    encodedFeatureNames.push(col);
                    totalEncodedFeatures += 1;
                }
            });

            // 3. Transform Rows into Matrix X and Vector y
            const X = [];
            const y = [];

            rows.forEach(row => {
                const xRow = new Array(totalEncodedFeatures).fill(0);
                featureCols.forEach(col => {
                    const meta = featureMetadata[col];
                    const val = row[col];

                    if (meta.type === 'nominal') {
                        const catIdx = meta.categories.indexOf(String(val));
                        if (catIdx !== -1) {
                            xRow[meta.startIndex + catIdx] = 1;
                        }
                    } else if (meta.type === 'ordinal') {
                        const rankVal = meta.rankMap[String(val)] !== undefined ? meta.rankMap[String(val)] : meta.defaultRank;
                        xRow[meta.startIndex] = rankVal;
                    } else {
                        // Continuous: standardized (missing -> training mean, i.e. 0 after scaling)
                        const num = this.isMissing(val) ? NaN : Number(val);
                        const scaled = (num - meta.mean) / meta.std;
                        xRow[meta.startIndex] = isNaN(scaled) ? 0 : Number(scaled.toFixed(5));
                    }
                });

                X.push(xRow);

                if (isClassification) {
                    const targetClass = String(row[targetCol]);
                    y.push(targetClassToIdx[targetClass] !== undefined ? targetClassToIdx[targetClass] : 0);
                } else {
                    y.push(Number(row[targetCol]));
                }
            });

            return {
                X,
                y,
                featureCols,
                targetCol,
                targetConf,
                isClassification,
                targetClasses,
                targetClassToIdx,
                targetIdxToClass,
                targetStats,
                featureMetadata,
                encodedFeatureNames,
                totalDimensions: totalEncodedFeatures,
                droppedMissingTarget,
                // Binary metrics (AUC, sensitivity, TP/FN) treat class index 1 as the positive class
                positiveClass: isClassification && targetClasses.length === 2 ? targetClasses[1] : null
            };
        },

        /**
         * Transform a single arbitrary row (for single-case or batch inference).
         * Missing or unrecognised values are replaced with training-derived values
         * (continuous: training mean; ordinal: median training rank; nominal: most frequent
         * training category). If an `issues` array is supplied, each substitution is recorded
         * so the caller can warn the user.
         */
        transformSingleRow(inputRow, prepMeta, issues = null) {
            const { featureCols, featureMetadata, totalDimensions } = prepMeta;
            const xRow = new Array(totalDimensions).fill(0);
            const note = (col, kind, rawVal) => {
                if (issues) issues.push({ column: col, kind, value: rawVal });
            };

            featureCols.forEach(col => {
                const meta = featureMetadata[col];
                const rawVal = inputRow[col];
                const missing = this.isMissing(rawVal);

                if (meta.type === 'nominal') {
                    let catIdx = missing ? -1 : meta.categories.indexOf(String(rawVal).trim());
                    if (catIdx === -1) {
                        note(col, missing ? 'missing' : 'unseen', rawVal);
                        catIdx = meta.categories.indexOf(meta.mode !== undefined ? meta.mode : meta.categories[0]);
                    }
                    if (catIdx !== -1) {
                        xRow[meta.startIndex + catIdx] = 1;
                    }
                } else if (meta.type === 'ordinal') {
                    let rankVal = missing ? undefined : meta.rankMap[String(rawVal).trim()];
                    if (rankVal === undefined && !missing && !isNaN(Number(rawVal))) {
                        rankVal = meta.rankMap[String(Number(rawVal))];
                    }
                    if (rankVal === undefined) {
                        note(col, missing ? 'missing' : 'unseen', rawVal);
                        rankVal = meta.defaultRank !== undefined ? meta.defaultRank : 0;
                    }
                    xRow[meta.startIndex] = rankVal;
                } else {
                    // Note: Number('') === 0, so blanks must be detected explicitly
                    const num = missing ? NaN : Number(rawVal);
                    if (isNaN(num)) {
                        note(col, missing ? 'missing' : 'non-numeric', rawVal);
                        xRow[meta.startIndex] = 0; // training mean (0 after standardisation)
                    } else {
                        const scaled = (num - meta.mean) / meta.std;
                        xRow[meta.startIndex] = isNaN(scaled) ? 0 : Number(scaled.toFixed(5));
                    }
                }
            });

            return xRow;
        },

        /**
         * Pseudo-Random Number Generator (Mulberry32) for reproducible data splits & shuffling
         */
        createPRNG(seed) {
            if (seed === null || seed === undefined || seed === '' || isNaN(seed)) {
                return Math.random;
            }
            let s = Math.floor(Math.abs(Number(seed))) >>> 0;
            if (s === 0) s = 1;
            return function() {
                s |= 0; s = (s + 0x6D2B79F5) | 0;
                let t = Math.imul(s ^ (s >>> 15), 1 | s);
                t = t + Math.imul(t ^ (t >>> 7), 61 | t) ^ t;
                return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
            };
        },

        /**
         * Train / Test Split with deterministic random seed support
         */
        trainTestSplit(X, y, trainRatio = 0.8, isClassification = true, seed = null) {
            const total = X.length;
            const indices = Array.from({ length: total }, (_, i) => i);
            const rng = this.createPRNG(seed);

            // Deterministic seeded Fisher-Yates shuffle
            for (let i = indices.length - 1; i > 0; i--) {
                const j = Math.floor(rng() * (i + 1));
                [indices[i], indices[j]] = [indices[j], indices[i]];
            }

            let trainIndices = [];
            let testIndices = [];

            if (isClassification) {
                // Stratified split: partition by class
                const classIndices = {};
                indices.forEach(idx => {
                    const cls = y[idx];
                    if (!classIndices[cls]) classIndices[cls] = [];
                    classIndices[cls].push(idx);
                });

                Object.keys(classIndices).forEach(cls => {
                    const cIdxs = classIndices[cls];
                    const trainCount = Math.max(1, Math.round(cIdxs.length * trainRatio));
                    trainIndices.push(...cIdxs.slice(0, trainCount));
                    testIndices.push(...cIdxs.slice(trainCount));
                });
            } else {
                const trainCount = Math.round(total * trainRatio);
                trainIndices = indices.slice(0, trainCount);
                testIndices = indices.slice(trainCount);
            }

            // Ensure test has at least 1 sample if total >= 4
            if (testIndices.length === 0 && total >= 4) {
                testIndices.push(trainIndices.pop());
            }

            return {
                trainX: trainIndices.map(i => X[i]),
                trainY: trainIndices.map(i => y[i]),
                testX: testIndices.map(i => X[i]),
                testY: testIndices.map(i => y[i]),
                trainIndices,
                testIndices
            };
        },

        /**
         * Transforms a separate test dataset using training metadata,
         * extracting X_test and y_test strictly without data leakage.
         */
        prepareTestingDataset(testRows, prepMeta) {
            const { targetCol, isClassification, targetClassToIdx } = prepMeta;
            const X_test = [];
            const y_test = [];
            const validRows = [];
            let missingTargetCount = 0;
            let unknownClassCount = 0;
            const featureIssues = [];

            testRows.forEach(row => {
                const rawTarget = row[targetCol];
                if (this.isMissing(rawTarget)) {
                    missingTargetCount++;
                    return;
                }

                let targetVal;
                if (isClassification) {
                    const strTarget = String(rawTarget).trim();
                    if (targetClassToIdx[strTarget] === undefined) {
                        unknownClassCount++;
                        return;
                    }
                    targetVal = targetClassToIdx[strTarget];
                } else {
                    targetVal = Number(rawTarget);
                    if (isNaN(targetVal)) {
                        missingTargetCount++;
                        return;
                    }
                }

                const xRow = this.transformSingleRow(row, prepMeta, featureIssues);
                X_test.push(xRow);
                y_test.push(targetVal);
                validRows.push(row);
            });

            return {
                X: X_test,
                y: y_test,
                validRows,
                totalEvaluated: X_test.length,
                totalProvided: testRows.length,
                missingTargetCount,
                unknownClassCount,
                // Predictor cells that were missing/unrecognised and filled from training statistics
                imputedFeatureCells: featureIssues.length
            };
        },

        /**
         * =========================================================================
         * 7-STEP PIPELINE EXTENSIONS
         * Strictly ordered, leak-free statistical & machine learning pipeline
         * Conceived & designed by Dr G Narenthiran MB ChB BSc(MedSci)(Hons) FEBNS FRCS(SN)
         * =========================================================================
         */

        /**
         * Fits an imputer strictly on the training set only (zero data leakage)
         */
        fitStrictImputer(trainRows, featureCols, targetCol, columnConfigs, strategy = 'median') {
            const fillValues = {};
            const columnStats = {};

            // Predictors only: the outcome (targetCol) is never imputed. Rows with a missing
            // outcome must be excluded before calling this (see executePipelineStep1).
            const allCols = featureCols.filter(c => c !== targetCol);

            allCols.forEach(col => {
                const conf = columnConfigs[col] || { type: 'continuous' };
                const validVals = [];
                const validNums = [];

                trainRows.forEach(r => {
                    const val = r[col];
                    if (!this.isMissing(val)) {
                        validVals.push(val);
                        const n = Number(val);
                        if (!isNaN(n)) validNums.push(n);
                    }
                });

                let fillVal = 0;
                const isTextOrdinal = conf.type === 'ordinal' && validVals.length > 0 && validNums.length < validVals.length;
                if (conf.type === 'nominal' || isTextOrdinal) {
                    // Mode for nominal
                    const freq = {};
                    let maxCount = 0;
                    let topVal = 'Unknown';
                    validVals.forEach(v => {
                        const s = String(v).trim();
                        freq[s] = (freq[s] || 0) + 1;
                        if (freq[s] > maxCount) {
                            maxCount = freq[s];
                            topVal = s;
                        }
                    });
                    fillVal = topVal;
                    columnStats[col] = { mode: topVal, distinctCount: Object.keys(freq).length };
                } else {
                    // Continuous / numeric Ordinal: Median or Mean (ordinal always uses the median, rounded to a valid level)
                    if (validNums.length > 0) {
                        const sum = validNums.reduce((a, b) => a + b, 0);
                        const mean = sum / validNums.length;
                        const sorted = [...validNums].sort((a, b) => a - b);
                        const mid = Math.floor(sorted.length / 2);
                        const median = sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
                        
                        if (conf.type === 'ordinal') {
                            // Lower median keeps the fill value on an observed ordinal level
                            fillVal = sorted[Math.floor((sorted.length - 1) / 2)];
                        } else {
                            fillVal = (strategy === 'mean') ? Number(mean.toFixed(4)) : Number(median.toFixed(4));
                        }
                        const variance = validNums.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / validNums.length;
                        columnStats[col] = {
                            mean: Number(mean.toFixed(4)),
                            median: Number(median.toFixed(4)),
                            std: Number(Math.sqrt(variance).toFixed(4)),
                            min: sorted[0],
                            max: sorted[sorted.length - 1],
                            validCount: validNums.length
                        };
                    } else {
                        fillVal = 0;
                        columnStats[col] = { mean: 0, median: 0, std: 1, min: 0, max: 0, validCount: 0 };
                    }
                }

                fillValues[col] = fillVal;
            });

            return {
                fillValues,
                columnStats,
                strategy,
                fittedSampleCount: trainRows.length
            };
        },

        /**
         * Transforms dataset rows using a pre-fitted strict imputer
         */
        transformWithStrictImputer(rows, imputer, columnConfigs) {
            let missingFilledCount = 0;
            const imputedRows = rows.map(r => {
                const newRow = { ...r };
                Object.keys(imputer.fillValues).forEach(col => {
                    if (this.isMissing(newRow[col])) {
                        newRow[col] = imputer.fillValues[col];
                        missingFilledCount++;
                    } else {
                        const conf = columnConfigs[col];
                        if (conf && (conf.type === 'continuous' || conf.type === 'ordinal')) {
                            const n = Number(newRow[col]);
                            if (!isNaN(n)) newRow[col] = n;
                        }
                    }
                });
                return newRow;
            });

            return {
                rows: imputedRows,
                missingFilledCount
            };
        },

        /**
         * Computes sample skewness to test for normality
         * g1 = [n / ((n-1)(n-2))] * sum(((x - mean)/s)^3)
         */
        calculateSkewness(arr) {
            const nums = arr.map(Number).filter(n => !isNaN(n));
            const n = nums.length;
            if (n < 3) return { skewness: 0, isNormal: true, mean: 0, std: 1 };

            const mean = nums.reduce((a, b) => a + b, 0) / n;
            const s2 = nums.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (n - 1);
            const s = Math.sqrt(s2);
            if (s === 0) return { skewness: 0, isNormal: true, mean, std: 0 };

            const m3 = nums.reduce((a, b) => a + Math.pow(b - mean, 3), 0);
            const g1 = (n * m3) / ((n - 1) * (n - 2) * Math.pow(s, 3));
            const isNormal = Math.abs(g1) <= 1.0;

            return {
                skewness: Number(g1.toFixed(3)),
                isNormal,
                mean: Number(mean.toFixed(3)),
                std: Number(s.toFixed(3))
            };
        },

        testNormality(arr) {
            return this.calculateSkewness(arr);
        },

        /**
         * Pearson Product-Moment Correlation Coefficient
         */
        calculatePearson(x, y) {
            const n = Math.min(x.length, y.length);
            if (n < 2) return 0;
            const xMean = x.reduce((a, b) => a + b, 0) / n;
            const yMean = y.reduce((a, b) => a + b, 0) / n;

            let num = 0;
            let denX = 0;
            let denY = 0;

            for (let i = 0; i < n; i++) {
                const dx = x[i] - xMean;
                const dy = y[i] - yMean;
                num += dx * dy;
                denX += dx * dx;
                denY += dy * dy;
            }

            const den = Math.sqrt(denX * denY);
            if (den === 0) return 0;
            return Number((num / den).toFixed(4));
        },

        /**
         * Spearman Rank-Order Correlation Coefficient
         */
        calculateSpearman(x, y) {
            const n = Math.min(x.length, y.length);
            if (n < 2) return 0;

            function getRanks(arr) {
                const indexed = arr.map((val, idx) => ({ val, idx }));
                indexed.sort((a, b) => a.val - b.val);
                const ranks = new Array(n);
                let i = 0;
                while (i < n) {
                    let j = i;
                    while (j < n - 1 && indexed[j + 1].val === indexed[j].val) {
                        j++;
                    }
                    const avgRank = (i + j + 2) / 2; // 1-based average rank
                    for (let k = i; k <= j; k++) {
                        ranks[indexed[k].idx] = avgRank;
                    }
                    i = j + 1;
                }
                return ranks;
            }

            const xRanks = getRanks(x);
            const yRanks = getRanks(y);
            return this.calculatePearson(xRanks, yRanks);
        },

        /**
         * Checks pairwise continuous variable correlation using Training set
         * Applies Pearson when both are normal; Spearman when one/both are skewed
         */
        checkPairwiseCorrelations(trainRows, continuousCols, threshold = 0.8) {
            const normalityMap = {};
            continuousCols.forEach(col => {
                const vals = trainRows.map(r => Number(r[col])).filter(n => !isNaN(n));
                normalityMap[col] = this.calculateSkewness(vals);
            });

            const pairs = [];
            for (let i = 0; i < continuousCols.length; i++) {
                for (let j = i + 1; j < continuousCols.length; j++) {
                    const cA = continuousCols[i];
                    const cB = continuousCols[j];
                    const valsA = trainRows.map(r => Number(r[cA]));
                    const valsB = trainRows.map(r => Number(r[cB]));

                    const normA = normalityMap[cA];
                    const normB = normalityMap[cB];
                    const bothNormal = normA.isNormal && normB.isNormal;
                    const method = bothNormal ? 'Pearson (r)' : 'Spearman (ρ)';
                    const coeff = bothNormal ? this.calculatePearson(valsA, valsB) : this.calculateSpearman(valsA, valsB);
                    const absCoeff = Math.abs(coeff);

                    pairs.push({
                        var1: cA,
                        var2: cB,
                        method,
                        coeff,
                        absCoeff,
                        norm1: normA,
                        norm2: normB,
                        bothNormal,
                        isHigh: absCoeff >= threshold
                    });
                }
            }

            // Sort descending by absolute correlation
            pairs.sort((a, b) => b.absCoeff - a.absCoeff);

            return {
                normalityMap,
                allPairs: pairs,
                highPairs: pairs.filter(p => p.isHigh)
            };
        },

        /**
         * Matrix Inversion using Gauss-Jordan elimination with partial pivoting
         */
        invertMatrix(A) {
            const n = A.length;
            const M = A.map((row, i) => {
                const r = [...row];
                const id = new Array(n).fill(0);
                id[i] = 1;
                return r.concat(id);
            });

            for (let i = 0; i < n; i++) {
                let maxRow = i;
                for (let k = i + 1; k < n; k++) {
                    if (Math.abs(M[k][i]) > Math.abs(M[maxRow][i])) maxRow = k;
                }
                [M[i], M[maxRow]] = [M[maxRow], M[i]];

                const pivot = M[i][i];
                if (Math.abs(pivot) < 1e-12) return null; // Singular matrix
                for (let j = 0; j < 2 * n; j++) M[i][j] /= pivot;

                for (let k = 0; k < n; k++) {
                    if (k === i) continue;
                    const factor = M[k][i];
                    for (let j = 0; j < 2 * n; j++) {
                        M[k][j] -= factor * M[i][j];
                    }
                }
            }
            return M.map(row => row.slice(n));
        },

        /**
         * Computes Variance Inflation Factor (VIF) on Training set features
         * One-hot encodes categoricals (dropping reference column) and computes R^2
         */
        calculateVIF(trainRows, featureCols, columnConfigs) {
            // Build design matrix with reference category dropped for nominal features
            const encodedCols = [];
            const colMapping = {}; // maps encoded column name back to original feature name

            featureCols.forEach(col => {
                const conf = columnConfigs[col];
                if (conf && conf.type === 'nominal') {
                    const distinct = Array.from(new Set(trainRows.map(r => String(r[col])))).sort();
                    // Drop reference column (first category) to avoid dummy trap
                    const nonRef = distinct.slice(1);
                    nonRef.forEach(cat => {
                        const dummyName = `${col}_is_${cat}`;
                        encodedCols.push({
                            name: dummyName,
                            originalCol: col,
                            type: 'dummy',
                            extract: (r) => String(r[col]) === cat ? 1 : 0
                        });
                    });
                } else {
                    // Continuous or Ordinal
                    encodedCols.push({
                        name: col,
                        originalCol: col,
                        type: 'numeric',
                        extract: (r) => {
                            const n = Number(r[col]);
                            return isNaN(n) ? 0 : n;
                        }
                    });
                }
            });

            if (encodedCols.length < 2) {
                return featureCols.map(col => ({
                    feature: col,
                    vif: 1.0,
                    rSquared: 0,
                    highVIF: false,
                    dummyCount: 1
                }));
            }

            const N = trainRows.length;
            const P = encodedCols.length;

            // Extract numeric values and standardize columns
            const X_raw = Array.from({ length: N }, () => new Array(P).fill(0));
            for (let j = 0; j < P; j++) {
                const extractor = encodedCols[j].extract;
                let sum = 0;
                for (let i = 0; i < N; i++) {
                    const val = extractor(trainRows[i]);
                    X_raw[i][j] = val;
                    sum += val;
                }
                const mean = sum / N;
                let varSum = 0;
                for (let i = 0; i < N; i++) varSum += Math.pow(X_raw[i][j] - mean, 2);
                const std = Math.sqrt(varSum / N) || 1;
                for (let i = 0; i < N; i++) {
                    X_raw[i][j] = (X_raw[i][j] - mean) / std;
                }
            }

            // Calculate VIF for each encoded column via regression on all other columns
            const encodedVIFs = [];
            for (let j = 0; j < P; j++) {
                // Dependent var y = X[:, j]
                const y = X_raw.map(row => row[j]);
                // Predictors Z = X[:, != j] with intercept
                const Z = X_raw.map(row => [1, ...row.filter((_, idx) => idx !== j)]);
                const dimZ = Z[0].length;

                // Compute Z^T * Z + lambda * I
                const ZtZ = Array.from({ length: dimZ }, () => new Array(dimZ).fill(0));
                const Zty = new Array(dimZ).fill(0);

                for (let i = 0; i < N; i++) {
                    const zRow = Z[i];
                    const yVal = y[i];
                    for (let a = 0; a < dimZ; a++) {
                        Zty[a] += zRow[a] * yVal;
                        for (let b = 0; b < dimZ; b++) {
                            ZtZ[a][b] += zRow[a] * zRow[b];
                        }
                    }
                }

                // Add small ridge lambda to guarantee well-conditioned inversion
                for (let a = 0; a < dimZ; a++) ZtZ[a][a] += 1e-4;

                const invZtZ = this.invertMatrix(ZtZ);
                let r2 = 0;
                if (invZtZ) {
                    const beta = new Array(dimZ).fill(0);
                    for (let a = 0; a < dimZ; a++) {
                        for (let b = 0; b < dimZ; b++) {
                            beta[a] += invZtZ[a][b] * Zty[b];
                        }
                    }

                    let ssTot = 0;
                    let ssRes = 0;
                    const yMean = y.reduce((a, b) => a + b, 0) / N;

                    for (let i = 0; i < N; i++) {
                        let yHat = 0;
                        for (let a = 0; a < dimZ; a++) yHat += Z[i][a] * beta[a];
                        ssTot += Math.pow(y[i] - yMean, 2);
                        ssRes += Math.pow(y[i] - yHat, 2);
                    }

                    r2 = ssTot > 0 ? Math.max(0, 1 - (ssRes / ssTot)) : 0;
                }

                let vif = r2 >= 0.999 ? 999.0 : 1 / Math.max(0.001, 1 - r2);
                encodedVIFs.push({
                    encodedName: encodedCols[j].name,
                    originalCol: encodedCols[j].originalCol,
                    rSquared: Number(r2.toFixed(4)),
                    vif: Number(vif.toFixed(2))
                });
            }

            // Aggregate back to original clinical feature level (taking the max VIF across its dummy levels)
            const featureVIFMap = {};
            encodedVIFs.forEach(item => {
                const col = item.originalCol;
                if (!featureVIFMap[col] || item.vif > featureVIFMap[col].vif) {
                    featureVIFMap[col] = {
                        feature: col,
                        vif: item.vif,
                        rSquared: item.rSquared,
                        type: columnConfigs[col]?.type || 'continuous',
                        dummyCount: (featureVIFMap[col]?.dummyCount || 0) + 1,
                        highVIF: item.vif > 5.0
                    };
                } else {
                    featureVIFMap[col].dummyCount++;
                }
            });

            const results = Object.values(featureVIFMap);
            results.sort((a, b) => b.vif - a.vif);
            return results;
        },

        /**
         * Accurate Lanczos approximation for log(Gamma(x))
         */
        logGamma(x) {
            const cof = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
            let y = x;
            let tmp = x + 5.5;
            tmp -= (x + 0.5) * Math.log(tmp);
            let ser = 1.000000000190015;
            for (let j = 0; j <= 5; j++) ser += cof[j] / ++y;
            return -tmp + Math.log(2.5066282746310005 * ser / x);
        },

        /**
         * Incomplete Gamma function P(a, x) for Chi-Square p-values
         */
        gammp(a, x) {
            if (x <= 0 || a <= 0) return 0;
            const ITMAX = 100, EPS = 3e-7, FPMIN = 1e-30;
            if (x < a + 1) {
                // Series representation
                let sum = 1 / a, del = sum, ap = a;
                for (let n = 1; n <= ITMAX; n++) {
                    ap += 1;
                    del *= x / ap;
                    sum += del;
                    if (Math.abs(del) < Math.abs(sum) * EPS) {
                        return sum * Math.exp(-x + a * Math.log(x) - this.logGamma(a));
                    }
                }
                return sum * Math.exp(-x + a * Math.log(x) - this.logGamma(a));
            } else {
                // Continued fraction representation
                let b = x + 1 - a, c = 1 / FPMIN, d = 1 / b, h = d;
                for (let i = 1; i <= ITMAX; i++) {
                    const an = -i * (i - a);
                    b += 2;
                    d = an * d + b;
                    if (Math.abs(d) < FPMIN) d = FPMIN;
                    c = b + an / c;
                    if (Math.abs(c) < FPMIN) c = FPMIN;
                    d = 1 / d;
                    const del = d * c;
                    h *= del;
                    if (Math.abs(del - 1) < EPS) break;
                }
                return 1 - (Math.exp(-x + a * Math.log(x) - this.logGamma(a)) * h);
            }
        },

        chiSquarePValue(chiSq, df) {
            if (chiSq <= 0 || df <= 0) return 1.0;
            const p = 1 - this.gammp(df / 2, chiSq / 2);
            return Math.max(0, Math.min(1, p));
        },

        /**
         * Incomplete Beta function I_x(a, b) for ANOVA / F-test p-values
         */
        betai(a, b, x) {
            if (x <= 0) return 0;
            if (x >= 1) return 1;
            const ITMAX = 100, EPS = 3e-7, FPMIN = 1e-30;
            const bt = Math.exp(this.logGamma(a + b) - this.logGamma(a) - this.logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));

            function betacf(aa, bb, xx) {
                const qab = aa + bb, qap = aa + 1, qam = aa - 1;
                let c = 1, d = 1 - qab * xx / qap;
                if (Math.abs(d) < FPMIN) d = FPMIN;
                d = 1 / d;
                let h = d;
                for (let m = 1; m <= ITMAX; m++) {
                    const m2 = 2 * m;
                    let num = m * (bb - m) * xx / ((qam + m2) * (aa + m2));
                    d = 1 + num * d;
                    if (Math.abs(d) < FPMIN) d = FPMIN;
                    c = 1 + num / c;
                    if (Math.abs(c) < FPMIN) c = FPMIN;
                    d = 1 / d;
                    h *= d * c;
                    num = -(aa + m) * (qab + m) * xx / ((aa + m2) * (qap + m2));
                    d = 1 + num * d;
                    if (Math.abs(d) < FPMIN) d = FPMIN;
                    c = 1 + num / c;
                    if (Math.abs(c) < FPMIN) c = FPMIN;
                    d = 1 / d;
                    const del = d * c;
                    h *= del;
                    if (Math.abs(del - 1) < EPS) break;
                }
                return h;
            }

            if (x < (a + 1) / (a + b + 2)) {
                return (bt * betacf(a, b, x)) / a;
            } else {
                return 1 - ((bt * betacf(b, a, 1 - x)) / b);
            }
        },

        fTestPValue(fStat, df1, df2) {
            if (fStat <= 0 || df1 <= 0 || df2 <= 0) return 1.0;
            const x = df2 / (df2 + df1 * fStat);
            return Math.max(0, Math.min(1, this.betai(df2 / 2, df1 / 2, x)));
        },

        /**
         * Univariate statistical baseline tests (Step 4)
         * - ANOVA F-test for continuous features vs categorical target
         * - Chi-Square test for categorical features vs categorical target
         * Returns summary table sorted by p-value (lowest to highest)
         */
        runUnivariateTesting(rows, survivingCols, targetCol, columnConfigs) {
            const isTargetContinuous = columnConfigs[targetCol]?.type === 'continuous';
            const results = [];
            // Exclude rows with a missing outcome from every test
            rows = rows.filter(r => !this.isMissing(r[targetCol]) && (!isTargetContinuous || !isNaN(Number(r[targetCol]))));

            // One-way ANOVA on numeric values grouped by a categorical key
            const oneWayAnova = (pairs) => {
                const groups = {};
                pairs.forEach(({ g, v }) => {
                    if (!groups[g]) groups[g] = [];
                    groups[g].push(v);
                });
                const groupKeys = Object.keys(groups).filter(k => groups[k].length > 0);
                const K = groupKeys.length;
                const N = pairs.length;
                const grandMean = N > 0 ? pairs.reduce((a, p) => a + p.v, 0) / N : 0;
                let ssBetween = 0;
                let ssWithin = 0;
                groupKeys.forEach(k => {
                    const grp = groups[k];
                    const grpMean = grp.reduce((a, b) => a + b, 0) / grp.length;
                    ssBetween += grp.length * Math.pow(grpMean - grandMean, 2);
                    grp.forEach(v => { ssWithin += Math.pow(v - grpMean, 2); });
                });
                const df1 = K - 1;
                const df2 = N - K;
                if (df1 < 1 || df2 < 1) {
                    return { fStat: 0, pVal: 1, df1: Math.max(0, df1), df2: Math.max(0, df2) };
                }
                const msBetween = ssBetween / df1;
                const msWithin = ssWithin / df2;
                const fStat = msWithin > 0 ? (msBetween / msWithin) : (msBetween > 0 ? Infinity : 0);
                const pVal = fStat === Infinity ? 0 : this.fTestPValue(fStat, df1, df2);
                return { fStat, pVal, df1, df2 };
            };

            // Numeric value of a feature cell (ordinal text levels are converted to their ordered rank)
            const numericValueFn = (col) => {
                const conf = columnConfigs[col] || {};
                const vals = rows.map(r => r[col]).filter(v => !this.isMissing(v));
                const isTextOrdinal = conf.type === 'ordinal' && vals.some(v => isNaN(Number(v)));
                if (isTextOrdinal) {
                    const { rankMap } = this.buildOrdinalRankMap(vals);
                    return (v) => (this.isMissing(v) || rankMap[String(v)] === undefined ? NaN : rankMap[String(v)]);
                }
                return (v) => (this.isMissing(v) ? NaN : Number(v));
            };

            const fmtStat = (v) => (v === Infinity ? Infinity : Number(v.toFixed(3)));

            survivingCols.forEach(col => {
                const isFeatureContinuous = columnConfigs[col]?.type === 'continuous' || columnConfigs[col]?.type === 'ordinal';

                if (!isTargetContinuous) {
                    // Categorical Target (e.g. Binary Classification)
                    const targetClasses = Array.from(new Set(rows.map(r => String(r[targetCol])))).sort();
                    const K = targetClasses.length;

                    if (isFeatureContinuous) {
                        // One-Way ANOVA F-test (equivalent to the two-sample t-test when K = 2: F = t²)
                        const toNum = numericValueFn(col);
                        const pairs = [];
                        rows.forEach(r => {
                            const v = toNum(r[col]);
                            if (!isNaN(v)) pairs.push({ g: String(r[targetCol]), v });
                        });
                        const a = oneWayAnova(pairs);

                        results.push({
                            feature: col,
                            scale: columnConfigs[col]?.type === 'ordinal' ? 'Ordinal (ranks)' : 'Continuous',
                            testName: K === 2 ? "Two-Sample t-test (as ANOVA F = t²)" : "One-Way ANOVA (F-test)",
                            statisticLabel: 'F-stat',
                            statisticValue: fmtStat(a.fStat),
                            df: `${a.df1}, ${a.df2}`,
                            pValue: Number(a.pVal.toPrecision(4)),
                            significant: a.pVal < 0.05
                        });
                    } else {
                        // Categorical vs Categorical: Chi-Square Test
                        const validRows = rows.filter(r => !this.isMissing(r[col]));
                        const categories = Array.from(new Set(validRows.map(r => String(r[col])))).sort();
                        const R = categories.length;

                        // Contingency Table
                        const obs = Array.from({ length: R }, () => new Array(K).fill(0));
                        const rowTotals = new Array(R).fill(0);
                        const colTotals = new Array(K).fill(0);
                        let totalN = 0;

                        validRows.forEach(r => {
                            const rIdx = categories.indexOf(String(r[col]));
                            const cIdx = targetClasses.indexOf(String(r[targetCol]));
                            if (rIdx !== -1 && cIdx !== -1) {
                                obs[rIdx][cIdx]++;
                                rowTotals[rIdx]++;
                                colTotals[cIdx]++;
                                totalN++;
                            }
                        });

                        let chiSq = 0;
                        let lowExpected = 0;
                        for (let i = 0; i < R; i++) {
                            for (let j = 0; j < K; j++) {
                                const expected = (rowTotals[i] * colTotals[j]) / (totalN || 1);
                                if (expected < 5) lowExpected++;
                                if (expected > 0) {
                                    chiSq += Math.pow(obs[i][j] - expected, 2) / expected;
                                }
                            }
                        }

                        const df = (R - 1) * (K - 1);
                        const pVal = df >= 1 ? this.chiSquarePValue(chiSq, df) : 1;
                        const cells = R * K;

                        results.push({
                            feature: col,
                            scale: 'Nominal Categorical',
                            testName: "Pearson Chi-Square (χ²)" + (lowExpected > 0 && cells > 0 && lowExpected / cells > 0.2
                                ? ' ⚠️ >20% of expected counts < 5: approximation unreliable' : ''),
                            statisticLabel: 'χ² stat',
                            statisticValue: Number(chiSq.toFixed(3)),
                            df: `${df}`,
                            pValue: Number(pVal.toPrecision(4)),
                            significant: pVal < 0.05
                        });
                    }
                } else {
                    // Continuous Target (Regression)
                    if (isFeatureContinuous) {
                        const toNum = numericValueFn(col);
                        const valsX = [];
                        const valsY = [];
                        rows.forEach(r => {
                            const x = toNum(r[col]);
                            const y = Number(r[targetCol]);
                            if (!isNaN(x) && !isNaN(y)) {
                                valsX.push(x);
                                valsY.push(y);
                            }
                        });
                        const n = valsX.length;
                        const rCoeff = this.calculatePearson(valsX, valsY);
                        let pVal = 1;
                        let tStat = 0;
                        if (n > 2) {
                            if (Math.abs(rCoeff) >= 1) {
                                tStat = rCoeff > 0 ? Infinity : -Infinity;
                                pVal = 0;
                            } else {
                                tStat = rCoeff * Math.sqrt((n - 2) / (1 - rCoeff * rCoeff));
                                pVal = this.fTestPValue(tStat * tStat, 1, n - 2);
                            }
                        }

                        results.push({
                            feature: col,
                            scale: columnConfigs[col]?.type === 'ordinal' ? 'Ordinal (ranks)' : 'Continuous',
                            testName: "Pearson Linear Correlation (t-test)",
                            statisticLabel: `r (t = ${Number.isFinite(tStat) ? tStat.toFixed(2) : '∞'})`,
                            statisticValue: Number(rCoeff.toFixed(3)),
                            df: `${Math.max(0, n - 2)}`,
                            pValue: Number(pVal.toPrecision(4)),
                            significant: pVal < 0.05
                        });
                    } else {
                        // Categorical feature vs Continuous Target: One-Way ANOVA
                        const pairs = [];
                        rows.forEach(r => {
                            if (!this.isMissing(r[col])) pairs.push({ g: String(r[col]), v: Number(r[targetCol]) });
                        });
                        const a = oneWayAnova(pairs);

                        results.push({
                            feature: col,
                            scale: 'Nominal Categorical',
                            testName: a.df1 === 1 ? "Two-Sample t-test (as ANOVA F = t²)" : "One-Way ANOVA (F-test)",
                            statisticLabel: 'F-stat',
                            statisticValue: fmtStat(a.fStat),
                            df: `${a.df1}, ${a.df2}`,
                            pValue: Number(a.pVal.toPrecision(4)),
                            significant: a.pVal < 0.05
                        });
                    }
                }
            });

            // Sort ascending by p-value (lowest to highest) as specified in Step 4
            results.sort((a, b) => a.pValue - b.pValue);
            return results;
        },

        /**
         * Step 5: Imbalance Handling using SMOTENC
         * Applied strictly to the Training set ONLY when the cohort is imbalanced:
         * the smallest class is < 30% of the training set for a binary outcome, or, for
         * multi-class outcomes, smaller than 30/70 (≈ 43%) of the largest class (the same
         * ratio as the binary 30% rule). Every under-represented class is oversampled up to
         * the size of the largest class.
         */
        applySMOTENC(trainRows, featureCols, targetCol, columnConfigs, seed = 42) {
            const isClassification = columnConfigs[targetCol]?.type !== 'continuous';
            if (!isClassification) {
                return {
                    balancedRows: trainRows,
                    applied: false,
                    reason: 'Target is continuous (Regression); SMOTENC applies to classification.'
                };
            }

            const classCounts = {};
            trainRows.forEach(r => {
                const c = String(r[targetCol]);
                classCounts[c] = (classCounts[c] || 0) + 1;
            });

            const classes = Object.keys(classCounts);
            if (classes.length < 2) {
                return {
                    balancedRows: trainRows,
                    applied: false,
                    classCounts,
                    reason: 'Single class present in training data.'
                };
            }

            const total = trainRows.length;
            let minorityClass = classes[0];
            let majorityClass = classes[0];
            let minCount = Infinity;
            let maxCount = -Infinity;

            classes.forEach(c => {
                if (classCounts[c] < minCount) {
                    minCount = classCounts[c];
                    minorityClass = c;
                }
                if (classCounts[c] > maxCount) {
                    maxCount = classCounts[c];
                    majorityClass = c;
                }
            });

            const minorityPct = minCount / total;
            const isBinary = classes.length === 2;
            const ratioThreshold = 0.30 / 0.70;
            const imbalanced = isBinary ? minorityPct < 0.30 : (minCount / maxCount) < ratioThreshold;
            const ruleText = isBinary
                ? '30% minority-share threshold'
                : `minority/majority ratio threshold of ${(ratioThreshold * 100).toFixed(0)}%`;

            if (!imbalanced) {
                return {
                    balancedRows: trainRows,
                    applied: false,
                    minorityClass,
                    minorityPct: Number((minorityPct * 100).toFixed(1)),
                    classCounts,
                    reason: isBinary
                        ? `Minority class is ${(minorityPct * 100).toFixed(1)}% (≥ 30% threshold). No synthetic resampling applied.`
                        : `Smallest class (${minCount}) is ${((minCount / maxCount) * 100).toFixed(0)}% of the largest (${maxCount}), within the ${ruleText}. No synthetic resampling applied.`
                };
            }

            // Continuous features and numeric ordinals are interpolated (ordinals then snapped to an observed level);
            // nominal features and text ordinals are treated as categorical.
            const isNumericCol = (c) => trainRows.every(r => !isNaN(Number(r[c])) && String(r[c]).trim() !== '');
            const contCols = featureCols.filter(c => columnConfigs[c]?.type === 'continuous' ||
                (columnConfigs[c]?.type === 'ordinal' && isNumericCol(c)));
            const catCols = featureCols.filter(c => !contCols.includes(c));
            const ordinalLevels = {};
            contCols.forEach(c => {
                if (columnConfigs[c]?.type === 'ordinal') {
                    ordinalLevels[c] = Array.from(new Set(trainRows.map(r => Number(r[c])))).sort((a, b) => a - b);
                }
            });

            const rng = this.createPRNG(seed);
            const syntheticRows = [];
            const generatedPerClass = {};
            const skippedClasses = [];

            classes.forEach(cls => {
                const needed = maxCount - classCounts[cls];
                if (needed <= 0) return;
                const minorityRows = trainRows.filter(r => String(r[targetCol]) === cls);
                const m = minorityRows.length;
                if (m < 2) {
                    skippedClasses.push(cls);
                    return;
                }

                // Standard deviations of continuous features within this class; distances use standardised units
                const stdMap = {};
                contCols.forEach(col => {
                    const vals = minorityRows.map(r => Number(r[col])).filter(n => !isNaN(n));
                    const mean = vals.reduce((a, b) => a + b, 0) / (vals.length || 1);
                    const v = vals.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (vals.length || 1);
                    const s = Math.sqrt(v);
                    stdMap[col] = s > 0 ? s : 1.0;
                });
                // SMOTENC penalises a categorical mismatch by the squared median std of the continuous features.
                // Because continuous differences are already divided by their std, that median std is 1 on this scale.
                const catPenalty = 1.0;

                const k = Math.min(5, m - 1);
                for (let sIdx = 0; sIdx < needed; sIdx++) {
                    const i = Math.floor(rng() * m);
                    const baseRow = minorityRows[i];

                    const dists = [];
                    for (let j = 0; j < m; j++) {
                        if (j === i) continue;
                        const otherRow = minorityRows[j];
                        let distSq = 0;
                        contCols.forEach(col => {
                            const diff = (Number(baseRow[col]) - Number(otherRow[col])) / stdMap[col];
                            if (!isNaN(diff)) distSq += diff * diff;
                        });
                        catCols.forEach(col => {
                            if (String(baseRow[col]) !== String(otherRow[col])) distSq += catPenalty;
                        });
                        dists.push({ row: otherRow, dist: Math.sqrt(distSq) });
                    }

                    dists.sort((a, b) => a.dist - b.dist);
                    const neighbors = dists.slice(0, k);
                    const chosenNeighbor = neighbors[Math.floor(rng() * neighbors.length)].row;

                    const newRow = {};
                    const lambda = rng();
                    contCols.forEach(col => {
                        const v1 = Number(baseRow[col]);
                        const v2 = Number(chosenNeighbor[col]);
                        let v = v1 + lambda * (v2 - v1);
                        if (ordinalLevels[col]) {
                            // Snap to the nearest observed ordinal level
                            v = ordinalLevels[col].reduce((best, lvl) => (Math.abs(lvl - v) < Math.abs(best - v) ? lvl : best), ordinalLevels[col][0]);
                            newRow[col] = v;
                        } else {
                            newRow[col] = Number(v.toFixed(4));
                        }
                    });

                    // Categorical features: mode among the k neighbours (ties -> the base sample's value)
                    catCols.forEach(col => {
                        const votes = {};
                        neighbors.forEach(nbr => {
                            const v = String(nbr.row[col]);
                            votes[v] = (votes[v] || 0) + 1;
                        });
                        let bestCat = String(baseRow[col]);
                        let topVotes = votes[bestCat] || 0;
                        Object.keys(votes).forEach(v => {
                            if (votes[v] > topVotes) {
                                topVotes = votes[v];
                                bestCat = v;
                            }
                        });
                        newRow[col] = bestCat;
                    });

                    newRow[targetCol] = cls;
                    syntheticRows.push(newRow);
                }
                generatedPerClass[cls] = needed;
            });

            const syntheticCount = syntheticRows.length;
            if (syntheticCount === 0) {
                return {
                    balancedRows: trainRows,
                    applied: false,
                    minorityClass,
                    minorityPct: Number((minorityPct * 100).toFixed(1)),
                    classCounts,
                    reason: `Imbalance detected (${ruleText}) but too few minority samples (< 2) for SMOTENC neighbour search. No resampling applied.`
                };
            }

            const balancedRows = [...trainRows, ...syntheticRows];
            const newCounts = { ...classCounts };
            Object.keys(generatedPerClass).forEach(c => { newCounts[c] += generatedPerClass[c]; });

            const classList = Object.keys(generatedPerClass).map(c => `${c} (+${generatedPerClass[c]})`).join(', ');
            return {
                balancedRows,
                applied: true,
                minorityClass,
                minorityPctBefore: Number((minorityPct * 100).toFixed(1)),
                minorityPctAfter: Number((newCounts[minorityClass] / balancedRows.length * 100).toFixed(1)),
                originalCounts: classCounts,
                balancedCounts: newCounts,
                classCounts,
                syntheticCountGenerated: syntheticCount,
                skippedClasses,
                reason: `Smallest class was ${(minorityPct * 100).toFixed(1)}% of the training set (${ruleText} breached). SMOTENC synthesised ${syntheticCount} training instance(s): ${classList}.` +
                    (skippedClasses.length ? ` Classes with < 2 samples could not be oversampled: ${skippedClasses.join(', ')}.` : '')
            };
        }
    };

    window.DataEngine = DataEngine;
})(window);
