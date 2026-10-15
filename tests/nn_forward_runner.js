// Trains the app's neural network (real TensorFlow.js, CPU backend) and exports its architecture, trained
// weights and raw outputs so that nn_forward_check.py can recompute the forward pass independently.
// Usage: node tests/nn_forward_runner.js <input.json> <output.json>   (called by nn_forward_check.py)
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const dir = path.join(__dirname, '..');
global.window = globalThis;
global.self = globalThis;
global.require = require;
console.log = () => {};   // TF.js prints a "use tfjs-node" banner on load
console.warn = () => {};
vm.runInThisContext(fs.readFileSync(path.join(dir, 'tf.min.js'), 'utf8'));
vm.runInThisContext(fs.readFileSync(path.join(dir, 'models.js'), 'utf8'));

const [inputPath, outputPath] = process.argv.slice(2);
const input = JSON.parse(fs.readFileSync(inputPath, 'utf8'));

(async () => {
    await tf.setBackend('cpu');
    await tf.ready();
    const out = { tfjs: tf.version['tfjs-core'], backend: tf.getBackend(), cases: [] };
    for (const c of input.cases) {
        const isClf = c.task !== 'regression';
        const nn = new DeepNeuralNetwork(isClf, c.numClasses, c.config);
        await nn.fit(c.trainX, c.trainY);
        if (nn.usedFallback) throw new Error(`${c.name}: TF.js fell back to the linear model (${nn.fallbackReason})`);
        const model = nn.tfModel;
        // Architecture as TF.js itself reports it (not as we intended it)
        const layers = model.layers.map(l => {
            const cfg = l.getConfig();
            return {
                className: l.getClassName(),
                units: cfg.units,
                activation: cfg.activation,
                rate: cfg.rate,
                useBias: cfg.useBias,
                l2: cfg.kernelRegularizer ? cfg.kernelRegularizer.config.l2 : null,
                weights: l.getWeights().map(w => ({ shape: w.shape, values: Array.from(w.dataSync()) }))
            };
        });
        // Raw (unrounded) model outputs and the app's user-facing outputs (rounded probabilities / values)
        const raw = tf.tidy(() => Array.from(model.predict(tf.tensor2d(c.testX)).dataSync()));
        const appOutputs = nn.predictOutputs(c.testX).map(o => ({ pred: o.pred, probabilities: o.probabilities || null }));
        out.cases.push({ name: c.name, task: c.task, numClasses: c.numClasses, config: c.config, layers, raw, appOutputs,
            training: nn.trainingInfo });
        nn.dispose();
        process.stderr.write(`  app NN trained: ${c.name}\n`);
    }
    fs.writeFileSync(outputPath, JSON.stringify(out));
})().catch(e => {
    process.stderr.write(`nn_forward_runner failed: ${e && e.stack ? e.stack : e}\n`);
    process.exit(1);
});
