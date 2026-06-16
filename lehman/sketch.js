const DATA_URL = "https://acervos-digitais.github.io/met-faces-utils";
const LDATA_URL = "https://acervos-digitais.github.io/met-faces-lehman-data";

const id2obj = {};
const maskDefinitions = {};

let oimg, dimg, mimg;
let cObj;
let styleSel;

async function preload() {
  const landmarkRes = await fetch(`${LDATA_URL}/json/landmarks.json`);
  const landmarksObj = await landmarkRes.json();

  const masksRes = await fetch(`${DATA_URL}/json/mp_masks_definitions.json`);
  const masksObj = await masksRes.json();
  Object.assign(maskDefinitions, masksObj);

  setupDropDowns(landmarksObj["landmarks"]);
}

function extractFaceMasks(faces, definitions) {
  return faces.map(face => face.length < 1 ? [] : definitions.map(idx => face[idx]));
}

function setupDropDowns(objs) {
  styleSel = createSelect();
  styleSel.position(100, 10);
  styleSel.changed(drawImage);
  [ "A1", "A2", "A2b", "A3", "A4", "A5",
    "noise",
    "EYE_0", "EYE_1", "EYE_2", "EYE_2b", "EYE_3", "EYE_4",
    "EYE_2_M", "EYE_3_M", "EYE_4_M"];
  [ "A2b", "A1", "A4",
    "EYE_0", "EYE_1", "EYE_2", "EYE_2b",
    "EYE_2_M", "EYE_3_M"].forEach(option => styleSel.option(option));

  const imgSel = createSelect();
  imgSel.position(10, 10);
  imgSel.changed(updateImage);

  for (const obj of objs) {
    if (
      "faces" in obj &&
      "mp" in obj["faces"] &&
      obj["faces"]["mp"]["count"] > 0
    ) {
      imgSel.option(obj["objectID"]);
      id2obj[obj["objectID"]] = obj;
    }
  }

  cObj = objs[0];
  oimg = loadImage(`${LDATA_URL}/image/500/${cObj["objectID"]}.jpg`, drawImage);
}

function setup() {
  createCanvas(windowWidth - 2, windowHeight - 2);
  noLoop();
}

function updateImage(evt) {
  background(0);
  cObj = id2obj[evt.target.value];
  oimg = loadImage(`${LDATA_URL}/image/500/${cObj["objectID"]}.jpg`, drawImage);
}

function drawFaceMasks(img, masks, pg) {
  const [iw, ih] = [img.width, img.height];

  for (const mask of masks) {
    if (mask.length < 1) continue;

    mask.unshift(mask.at(0));
    mask.push(mask.at(0));
    mask.push(mask.at(1));

    pg.beginShape();
    for (const point of mask) {
      pg.curveVertex(point[0] * iw, point[1] * ih);
    }
    pg.endShape();
  }
}

function drawNoiseEye(center, radius, pg) {
  pg.push();
  pg.translate(center[0], center[1]);

  pg.beginShape();
  for (let a = 0; a < TWO_PI; a += PI / 100) {
    const x0 = radius * cos(a);
    const y0 = radius * sin(a);

    const rr = noise((pg.width + x0) / (3 * radius), (pg.height + y0) / (3 * radius), center[0]);

    vertex(rr * x0, rr * y0);
  }
  pg.endShape(CLOSE);
  pg.pop();
}

function drawNoiseMasks(img, faces, pg) {
  const [iw, ih] = [img.width, img.height];

  for (const face of faces) {
    if (face.length < 1) continue;
    noiseSeed(1010);

    const lCenter = face[473];
    const lOuter = face[446];
    const lRadius = (lOuter[0] - lCenter[0]) * iw;
    const rCenter = face[468];
    const rOuter = face[226];
    const rRadius = (rOuter[0] - rCenter[0]) * iw;

    const lrRadius = 2 * max(lRadius, rRadius);

    drawNoiseEye([lCenter[0] * iw, lCenter[1] * ih], lrRadius, pg);
    drawNoiseEye([rCenter[0] * iw, rCenter[1] * ih], lrRadius, pg);
  }
}

function createMask(img, mpObj, maskName) {
  const [iw, ih] = [img.width, img.height];

  const pg = createGraphics(iw, ih);
  pg.clear();
  pg.fill(255);
  pg.noStroke();

  if (maskName.toLowerCase().includes("noise")) {
    drawNoiseMasks(img, mpObj["landmarks"], pg);
  } else if (maskName in maskDefinitions) {
    const masks = extractFaceMasks(mpObj["landmarks"], maskDefinitions[maskName]);
    drawFaceMasks(img, masks, pg);
  } else {
    ["L", "R"].forEach(side => {
      const masks = extractFaceMasks(mpObj["landmarks"], maskDefinitions[`${maskName}_${side}`]);
      drawFaceMasks(img, masks, pg);
    });
  }

  return pg;
}

function drawImage() {
  background(0);
  const [oiw, oih] = [oimg.width, oimg.height];

  const minSF = min(width / oiw, height / oih);
  oimg.resize(minSF * oiw, minSF * oih);
  image(oimg, 0, 0);

  const [niw, nih] = [oimg.width, oimg.height];
  dimg = createImage(niw, nih);
  dimg.copy(oimg, 0, 0, niw, nih, 0, 0, niw, nih);

  mimg = createMask(oimg, cObj["faces"]["mp"], styleSel.value());
}

function mouseMoved() {
  if (!oimg || !dimg || !mimg) return;

  const inW = mouseX > 20 && mouseX < oimg.width - 20;
  const inH = mouseY > 20 && mouseY < oimg.height - 20;

  background(0);
  if (inW && inH) {
    dimg.mask(mimg);
    image(dimg, 0, 0);
    // image(mimg, 0, 0);
  } else {
    image(oimg, 0, 0);
  }
}
