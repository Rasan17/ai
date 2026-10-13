/**
 * NPH ML/Neuralnet Predictor - Data Engine
 * Handles CSV parsing, variable role & type deduction, missing value detection,
 * multi-strategy data imputation, feature encoding, scaling, and train/test splitting.
 * 
 * Conceived, designed and developed by Dr G Narenthiran MB ChB BSc(MedSci)(Hons) FEBNS FRCS(SN)
 */

(function (window) {
    'use strict';

    const DataEngine = {
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
                } else if (colIdx === headers.length - 1 || lowerCol.match(/^(target|outcome|response|shunt_response|label|class|diagnos|result|improvement)/i)) {
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

            // First pass: identify rows that must be dropped if user chose 'drop' on any column
            Object.keys(columnConfigs).forEach(col => {
                const config = columnConfigs[col];
                if (config.role === 'meta') return; // skip meta columns

                if (config.imputationStrategy === 'drop') {
                    workingRows.forEach((row, rIdx) => {
                        if (this.isMissing(row[col])) {
                            rowsToDrop.add(rIdx);
                        }
                    });
                }
            });

            if (rowsToDrop.size > 0) {
                workingRows = workingRows.filter((_, idx) => !rowsToDrop.has(idx));
                audit.push({
                    type: 'drop_rows',
                    description: `Dropped ${rowsToDrop.size} rows with missing values in columns marked for row deletion.`
                });
            }

            // Second pass: Compute imputation values on the filtered rows
            Object.keys(columnConfigs).forEach(col => {
                const config = columnConfigs[col];
                const analysis = columnAnalysis[col];
                if (config.role === 'meta') return;
                if (config.imputationStrategy === 'drop') return;

                const strategy = config.imputationStrategy;
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
                        fillValue = sorted.length % 2 === 0 ? Number(((sorted[mid - 1] + sorted[mid]) / 2).toFixed(4)) : sorted[mid];
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
                    const numericCols = Object.keys(columnConfigs).filter(c => columnConfigs[c].type === 'continuous' && c !== col);
                    workingRows.forEach((row, rowIdx) => {
                        if (this.isMissing(row[col])) {
                            // Find k nearest rows that have non-missing col
                            const k = 3;
                            const distances = [];
                            workingRows.forEach((otherRow, otherIdx) => {
                                if (otherIdx === rowIdx || this.isMissing(otherRow[col])) return;
                                let distSq = 0;
                                let matchedFeatures = 0;
                                numericCols.forEach(nCol => {
                                    if (!this.isMissing(row[nCol]) && !this.isMissing(otherRow[nCol])) {
                                        const d = Number(row[nCol]) - Number(otherRow[nCol]);
                                        distSq += d * d;
                                        matchedFeatures++;
                                    }
                                });
                                if (matchedFeatures > 0) {
                                    distances.push({ val: Number(otherRow[col]), dist: Math.sqrt(distSq) });
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
                        filledValue: fillValue === 'knn_dynamic' ? 'KNN Distance Weighted' : fillValue
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
                    min: Math.min(...nums),
                    max: Math.max(...nums)
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
                    rows.forEach(r => catSet.add(String(r[col])));
                    const categories = Array.from(catSet).sort();
                    featureMetadata[col] = {
                        type: 'nominal',
                        categories,
                        startIndex: totalEncodedFeatures,
                        length: categories.length
                    };
                    categories.forEach(cat => {
                        encodedFeatureNames.push(`${col}_${cat}`);
                    });
                    totalEncodedFeatures += categories.length;
                } else if (conf.type === 'ordinal') {
                    // Ordinal mapping: map distinct values to integers 0, 1, 2...
                    // Check if already numeric or string
                    const valSet = new Set();
                    rows.forEach(r => valSet.add(r[col]));
                    let ranks = Array.from(valSet);
                    const allNum = ranks.every(v => !isNaN(Number(v)));
                    if (allNum) {
                        ranks.sort((a, b) => Number(a) - Number(b));
                    } else {
                        ranks.sort();
                    }
                    const rankMap = {};
                    ranks.forEach((val, rk) => {
                        rankMap[String(val)] = rk;
                    });
                    featureMetadata[col] = {
                        type: 'ordinal',
                        ranks,
                        rankMap,
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
                        min: Math.min(...nums),
                        max: Math.max(...nums),
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
                        const rankVal = meta.rankMap[String(val)] !== undefined ? meta.rankMap[String(val)] : 0;
                        xRow[meta.startIndex] = rankVal;
                    } else {
                        // Continuous: standardized
                        const num = Number(val);
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
                totalDimensions: totalEncodedFeatures
            };
        },

        /**
         * Transform a single arbitrary row (for single-case or batch inference)
         */
        transformSingleRow(inputRow, prepMeta) {
            const { featureCols, featureMetadata, totalDimensions } = prepMeta;
            const xRow = new Array(totalDimensions).fill(0);

            featureCols.forEach(col => {
                const meta = featureMetadata[col];
                const rawVal = inputRow[col];

                if (meta.type === 'nominal') {
                    const catIdx = meta.categories.indexOf(String(rawVal));
                    if (catIdx !== -1) {
                        xRow[meta.startIndex + catIdx] = 1;
                    }
                } else if (meta.type === 'ordinal') {
                    const rankVal = meta.rankMap[String(rawVal)] !== undefined ? meta.rankMap[String(rawVal)] : 0;
                    xRow[meta.startIndex] = rankVal;
                } else {
                    const num = Number(rawVal);
                    if (isNaN(num)) {
                        xRow[meta.startIndex] = 0; // fallback to mean (which is 0 after standardisation)
                    } else {
                        const scaled = (num - meta.mean) / meta.std;
                        xRow[meta.startIndex] = isNaN(scaled) ? 0 : Number(scaled.toFixed(5));
                    }
                }
            });

            return xRow;
        },

        /**
         * Train / Test Split
         */
        trainTestSplit(X, y, trainRatio = 0.8, isClassification = true) {
            const total = X.length;
            const indices = Array.from({ length: total }, (_, i) => i);

            // Seedable or random shuffle
            for (let i = indices.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
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

                const xRow = this.transformSingleRow(row, prepMeta);
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
                unknownClassCount
            };
        }
    };

    window.DataEngine = DataEngine;
})(window);
