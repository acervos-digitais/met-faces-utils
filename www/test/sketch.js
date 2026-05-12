const DATA_URL = "https://acervos-digitais.github.io/met-faces-data";

const id2obj = {};

let img;
let oimg;
let maskGraphics;
let cObj;

async function preload() {
  const res = await fetch(`${DATA_URL}/json/landmarks.json`);
  const resObj = await res.json();
  setupDropDown(resObj["landmarks"]);
}

function setupDropDown(objs) {
  const mySelect = createSelect();
  mySelect.position(10, 10);
  mySelect.changed(updateImage);

  for (const obj of objs) {
    if (
      "faces" in obj &&
      "mp" in obj["faces"] &&
      obj["faces"]["mp"]["count"] > 0
    ) {
      mySelect.option(obj["objectID"]);
      id2obj[obj["objectID"]] = obj;
    }
  }
}

function setup() {
  createCanvas(windowWidth - 2, windowHeight - 2);
  noLoop();
}

function updateImage(evt) {
  background(0);
  cObj = id2obj[evt.target.value];
  oimg = loadImage(`${DATA_URL}/image/500/${cObj["objectID"]}.jpg`, drawImage);
}

function drawImage() {
  const minSF = min(width / oimg.width, height / oimg.height);
  oimg.resize(oimg.width * minSF, oimg.height * minSF);
  image(oimg, 0, 0);

  img = createImage(oimg.width, oimg.height);
  img.copy(oimg, 0, 0, oimg.width, oimg.height, 0, 0, oimg.width, oimg.height);

  const masks = cObj["faces"]["mp"]["masks"];

  maskGraphics = createGraphics(oimg.width, oimg.height);
  maskGraphics.clear();
  maskGraphics.fill(255);
  maskGraphics.noStroke();

  for (const mask of masks) {
    if (mask.length < 1) continue;

    maskGraphics.beginShape();
    maskGraphics.curveVertex(mask[0][0] * img.width, mask[0][1] * img.height);

    for (const point of mask) {
      maskGraphics.curveVertex(point[0] * img.width, point[1] * img.height);
    }

    maskGraphics.curveVertex(
      mask[mask.length - 1][0] * img.width,
      mask[mask.length - 1][1] * img.height,
    );

    maskGraphics.endShape(CLOSE);
  }
}

function mouseMoved() {
  if (!oimg || !img || !maskGraphics) return;

  const inW = mouseX > 20 && mouseX < oimg.width - 20;
  const inH = mouseY > 20 && mouseY < oimg.height - 20;

  background(0);
  if (inW && inH) {
    img.mask(maskGraphics);
    image(img, 0, 0);
  } else {
    image(oimg, 0, 0);
  }
}
