import * as tf from "@tensorflow/tfjs";
import { renderBoxes } from "./renderBox";
import labels from "./labels.json";

const numClass = labels.length;

/**
 * Preprocess image / frame before forwarded into the model
 * @param {HTMLVideoElement|HTMLImageElement} source
 * @param {Number} modelWidth
 * @param {Number} modelHeight
 * @returns input tensor, xRatio and yRatio
 */
const preprocess = (source, modelWidth, modelHeight) => {
  let xRatio, yRatio; // ratios for boxes

  const input = tf.tidy(() => {
    const img = tf.browser.fromPixels(source);

    // padding image to square => [n, m] to [n, n], n > m
    const [h, w] = img.shape.slice(0, 2); // get source width and height
    const maxSize = Math.max(w, h); // get max size
    const imgPadded = img.pad([
      [0, maxSize - h], // padding y [bottom only]
      [0, maxSize - w], // padding x [right only]
      [0, 0],
    ]);

    xRatio = maxSize / w; // update xRatio
    yRatio = maxSize / h; // update yRatio

    return tf.image
      .resizeBilinear(imgPadded, [modelWidth, modelHeight]) // resize frame
      .div(255.0) // normalize
      .expandDims(0); // add batch
  });

  return [input, xRatio, yRatio];
};

/**
 * Function to detect image.
 * @param {HTMLImageElement} imgSource image source
 * @param {tf.GraphModel} model loaded YOLOv8 tensorflow.js model
 * @param {Number} classThreshold class threshold
 * @param {HTMLCanvasElement} canvasRef canvas reference
 */
export const detectImage = async (imgSource, model, canvasRef) => {
  const [modelHeight, modelWidth] = model.inputShape.slice(1, 3); // get model width and height
  const modelSegChannel = model.outputShape[1][3];

  tf.engine().startScope(); // start scoping tf engine
  const [input, xRatio, yRatio] = preprocess(imgSource, modelWidth, modelHeight);

  const res = model.net.execute(input);
  const transRes = res[0].transpose([0, 2, 1]).squeeze();
  const transSegMask = res[1].transpose([0, 3, 1, 2]).squeeze();

  const boxes = tf.tidy(() => {
    const w = transRes.slice([0, 2], [-1, 1]);
    const h = transRes.slice([0, 3], [-1, 1]);
    const x1 = tf.sub(transRes.slice([0, 0], [-1, 1]), tf.div(w, 2)); //x1
    const y1 = tf.sub(transRes.slice([0, 1], [-1, 1]), tf.div(h, 2)); //y1
    return tf
      .concat(
        [
          y1,
          x1,
          tf.add(y1, h), //y2
          tf.add(x1, w), //x2
        ],
        1
      )
      .squeeze();
  });
  const masks = transRes.slice([0, 4], [-1, modelSegChannel]).squeeze();

  const [scores, classes] = tf.tidy(() => {
    const rawScores = transRes.slice([0, 4], [-1, numClass]).squeeze();
    return [rawScores.max(1), rawScores.argMax(1)];
  });

  const nms = await tf.image.nonMaxSuppressionAsync(boxes, scores, 500, 0.45, 0.2);

  const boxesToDraw = tf.tidy(() => {
    const toDraw = [];
    const overlay = [];

    const detReady = tf.concat(
      [
        boxes.gather(nms, 0),
        scores.gather(nms, 0).expandDims(1),
        classes.gather(nms, 0).expandDims(1),
        masks.gather(nms, 0),
      ],
      1
    );

    for (let i = 0; i < detReady.shape[0]; i++) {
      const [y1, x1, y2, x2] = detReady.slice([i, 0], [1, 4]).round().cast("int32").dataSync();
      const score = detReady.slice([i, 4], [1, 1]).dataSync();
      const label = detReady.slice([i, 5], [1, 1]).cast("int32").dataSync();
      const mask = detReady.slice([i, 6], [1, modelSegChannel]);

      // TODO: Fixing slicing protos
      //const cutProtos = transSegMask.slice([0, y1, x1], [-1, y2 - y1, x2 - x1]);

      //console.log(tf.matMul(mask, cutProtos.reshape([modelSegChannel, -1])));
      toDraw.push({
        box: [y1 * yRatio, x1 * xRatio, y2 * yRatio, x2 * xRatio],
        score: score[0],
        klass: label[0],
        label: labels[label[0]],
      });
    }

    return toDraw;
  });

  renderBoxes(canvasRef, boxesToDraw); // render boxes

  tf.dispose(res); // clear memory
  tf.dispose([transRes, transSegMask, boxes, scores, classes, nms]); // clear memory

  tf.engine().endScope(); // end of scoping
};

/**
 * Function to detect video from every source.
 * @param {HTMLVideoElement} vidSource video source
 * @param {tf.GraphModel} model loaded YOLOv8 tensorflow.js model
 * @param {Number} classThreshold class threshold
 * @param {HTMLCanvasElement} canvasRef canvas reference
 */
export const detectVideo = (vidSource, model, classThreshold, canvasRef) => {
  const [modelWidth, modelHeight] = model.inputShape.slice(1, 3); // get model width and height

  /**
   * Function to detect every frame from video
   */
  const detectFrame = async () => {
    if (vidSource.videoWidth === 0 && vidSource.srcObject === null) {
      const ctx = canvasRef.getContext("2d");
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // clean canvas
      return; // handle if source is closed
    }

    tf.engine().startScope(); // start scoping tf engine
    const [input, xRatio, yRatio] = preprocess(vidSource, modelWidth, modelHeight);

    const res = model.net.execute(input);
    const [boxes, scores, classes] = res.slice(0, 3);
    const boxes_data = boxes.dataSync();
    const scores_data = scores.dataSync();
    const classes_data = classes.dataSync();
    renderBoxes(canvasRef, classThreshold, boxes_data, scores_data, classes_data, [xRatio, yRatio]); // render boxes
    tf.dispose(res); // clear memory

    requestAnimationFrame(detectFrame); // get another frame
    tf.engine().endScope(); // end of scoping
  };

  detectFrame(); // initialize to detect every frame
};
