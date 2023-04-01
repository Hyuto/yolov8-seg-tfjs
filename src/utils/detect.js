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
 * @param {HTMLImageElement} source Source
 * @param {tf.GraphModel} model loaded YOLOv8 tensorflow.js model
 * @param {Number} classThreshold class threshold
 * @param {HTMLCanvasElement} canvasRef canvas reference
 * @param {VoidFunction} callback Callback function to run after detect frame is done
 */
export const detectFrame = async (source, model, canvasRef, callback = () => {}) => {
  const [modelHeight, modelWidth] = model.inputShape.slice(1, 3); // get model width and height
  const [modelSegHeight, modelSegWidth, modelSegChannel] = model.outputShape[1].slice(1);

  tf.engine().startScope(); // start scoping tf engine
  const [input, xRatio, yRatio] = preprocess(source, modelWidth, modelHeight);

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

  const [scores, classes] = tf.tidy(() => {
    const rawScores = transRes.slice([0, 4], [-1, numClass]).squeeze();
    return [rawScores.max(1), rawScores.argMax(1)];
  });
  const masks = transRes.slice([0, 4 + numClass], [-1, modelSegChannel]).squeeze();

  const nms = await tf.image.nonMaxSuppressionAsync(boxes, scores, 500, 0.45, 0.2);

  const boxesToDraw = tf.tidy(() => {
    const toDraw = [];
    let overlay = tf.zeros([modelHeight, modelWidth, 4]);

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
      const [y1, x1, y2, x2] = detReady.slice([i, 0], [1, 4]).dataSync();
      const score = detReady.slice([i, 4], [1, 1]).dataSync();
      const label = detReady.slice([i, 5], [1, 1]).cast("int32").dataSync();
      const mask = detReady.slice([i, 6], [1, modelSegChannel]);

      const upSampleBox = [
        Math.floor(y1 * yRatio), // y
        Math.floor(x1 * xRatio), // x
        Math.round((y2 - y1) * yRatio), // h
        Math.round((x2 - x1) * xRatio), // w
      ];
      const downSampleBox = [
        Math.floor((y1 * modelSegHeight) / modelHeight), // y
        Math.floor((x1 * modelSegWidth) / modelWidth), // x
        Math.round(((y2 - y1) * modelSegHeight) / modelHeight), // h
        Math.round(((x2 - x1) * modelSegWidth) / modelWidth), // w
      ];

      const cutProtos = transSegMask.slice(
        [
          0,
          downSampleBox[0] >= 0 ? downSampleBox[0] : 0,
          downSampleBox[1] >= 0 ? downSampleBox[1] : 0,
        ],
        [
          -1,
          downSampleBox[0] + downSampleBox[2] <= modelSegHeight
            ? downSampleBox[2]
            : modelSegHeight - downSampleBox[0],
          downSampleBox[1] + downSampleBox[3] <= modelSegWidth
            ? downSampleBox[3]
            : modelSegWidth - downSampleBox[1],
        ]
      );
      const protos = tf
        .matMul(mask, cutProtos.reshape([modelSegChannel, -1]))
        .reshape([downSampleBox[2], downSampleBox[3]])
        .expandDims(-1);
      const upsampleProtos = tf.image.resizeBilinear(protos, [upSampleBox[2], upSampleBox[3]]);
      const masked = tf.where(upsampleProtos.greaterEqual(0.5), [255, 255, 255, 150], [0, 0, 0, 0]);
      const maskedPaded = masked.pad([
        [upSampleBox[0], modelHeight - (upSampleBox[0] + upSampleBox[2])],
        [upSampleBox[1], modelWidth - (upSampleBox[1] + upSampleBox[3])],
        [0, 0],
      ]);

      overlay = addWeighted(overlay, maskedPaded, 1, 1);

      toDraw.push({
        box: upSampleBox,
        score: score[0],
        klass: label[0],
        label: labels[label[0]],
      });
    }

    const maskImg = new ImageData(
      new Uint8ClampedArray(overlay.cast("int32").dataSync()),
      modelHeight,
      modelWidth
    ); // create image data from mask overlay

    const ctx = canvasRef.getContext("2d");
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // clean canvas
    ctx.putImageData(maskImg, 0, 0);

    return toDraw;
  });

  renderBoxes(canvasRef, boxesToDraw); // render boxes

  tf.dispose(res); // clear memory
  tf.dispose([transRes, transSegMask, boxes, scores, classes, nms]); // clear memory

  callback();

  tf.engine().endScope(); // end of scoping
};

/**
 * Function to detect video from every source.
 * @param {HTMLVideoElement} vidSource video source
 * @param {tf.GraphModel} model loaded YOLOv8 tensorflow.js model
 * @param {HTMLCanvasElement} canvasRef canvas reference
 */
export const detectVideo = (vidSource, model, canvasRef) => {
  /**
   * Function to detect every frame from video
   */
  const detect = async () => {
    if (vidSource.videoWidth === 0 && vidSource.srcObject === null) {
      const ctx = canvasRef.getContext("2d");
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // clean canvas
      return; // handle if source is closed
    }

    detectFrame(vidSource, model, canvasRef, () => {
      requestAnimationFrame(detect); // get another frame
    });
  };

  detect(); // initialize to detect every frame
};

const addWeighted = (a, b, alpha, beta) => {
  return tf.tidy(() => {
    const combine = tf.add(tf.mul(a, alpha), tf.mul(b, beta));
    return combine.where(combine.lessEqual(255), 255);
  });
};
