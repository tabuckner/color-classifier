import * as tf from '@tensorflow/tfjs-node';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import { loadImage, createCanvas } from 'canvas';
import { createWriteStream, readdirSync } from 'fs';
import { join } from 'path';
const gm = require('gm');

let net;

const LABELS_ENUM = {
  black: 0,
  blue: 1,
  brown: 2,
  green: 3,
  orange: 4,
  red: 5,
  violet: 6,
  white: 7,
  yellow: 8
};

async function app() {
  console.log('Loading mobilenet');
  net = await mobilenet.load();
  const classifier = knnClassifier.create();

  const pictureDirectory = './color-images'
  const colorImageDir = readdirSync(pictureDirectory);

  for (const colorLabel of colorImageDir) {
    if (colorLabel !== '.DS_Store') {
      console.warn(`Adding predictions for ${colorLabel} to KNN.`);
    
      const labelDirPath = join(pictureDirectory, colorLabel);
      const images = readdirSync(labelDirPath);
  
      for (const image of images) {
        const filePath = join(labelDirPath, image);
        const imgCanvas = await createHtmlCanvasElement(filePath);
        const activation = await net.infer(imgCanvas);
        // // const testFilePath = join('./test', file);
        // // saveCreatedImage(imgCanvas, testFilePath);
        console.warn(`Adding Example for ${colorLabel} with file: ${image}`);
        classifier.addExample(activation, colorLabel);
      }
    }
    console.warn('Finished adding examples to KNN.')
  }

  const testImageCanvas = await createHtmlCanvasElement('./images/brown-training-example.jpg')
  await saveCreatedImage(testImageCanvas, 'test.jpg')
  const mobileNetClassification = await net.classify(testImageCanvas);
  const activation = await net.infer(testImageCanvas);
  const knnPrediction = await classifier.predictClass(activation)
  console.warn(knnPrediction)
}

async function createHtmlCanvasElement(filePath) {
  const destinationSize = 500;
  const canvas = createCanvas(destinationSize, destinationSize);
  const ctx = canvas.getContext('2d');
  const oldDimensions = await getSize(filePath);
  const oldWidth = oldDimensions.width;
  const oldHeight = oldDimensions.height;
  const img = await loadImage(filePath);
  const scalingFactor = oldWidth > oldHeight ? destinationSize / oldWidth : destinationSize / oldHeight;
  const newWidth = oldWidth * scalingFactor;
  const newHeight = oldHeight * scalingFactor;
  ctx.drawImage(img, 0, 0, newWidth, newHeight);
  return canvas;
}

async function saveCreatedImage(canvas, filename) {
  const out = createWriteStream(join(__dirname, filename));
  const stream = canvas.createJPEGStream();
  stream.pipe(out);
  out.on('finish', () => console.log(`${filename} saved`));
}

async function getSize(filename) {
  return new Promise((res, rej) => {
    gm(filename).size((err, size) => {
      if (err) {
        rej(err);
      }
      res(size);
    });
  });
}

app();