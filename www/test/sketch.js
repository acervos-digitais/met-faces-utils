const DATA_URL = "https://acervos-digitais.github.io/met-faces-data";

const id2obj = {};

let oimg, dimg, mimg;
let cObj;
let styleSel;

async function preload() {
  const res = await fetch(`${DATA_URL}/json/landmarks.json`);
  const resObj = await res.json();
  setupDropDowns(resObj["landmarks"]);
}

function setupDropDowns(objs) {
  styleSel = createSelect();
  styleSel.position(100, 10);
  styleSel.changed(drawImage);
  ["landmarks", "ovals"].forEach(option => styleSel.option(option));

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
  oimg = loadImage(`${DATA_URL}/image/500/${cObj["objectID"]}.jpg`, drawImage);
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

function createMasksMask(img, mpObj) {
  const [iw, ih] = [img.width, img.height];
  const masks = mpObj["eye_masks"];

  const pg = createGraphics(iw, ih);
  pg.clear();
  pg.fill(255);
  pg.noStroke();

  for (const mask of masks) {
    if (mask.length < 1) continue;

    pg.beginShape();

    for (const point of mask) {
      pg.curveVertex(point[0] * iw, point[1] * ih);
    }

    pg.curveVertex(mask[0][0] * iw, mask[0][1] * ih);
    pg.endShape();
  }
  return pg;
}

function createEllipsesMask(img, mpObj) {
  const [iw, ih] = [img.width, img.height];
  const centers = mpObj["eye_centers"];
  const corners = mpObj["eye_corners"];
  const xu = createVector(1,0);

  const vectorFromPct = (xy) => {
    return createVector(xy[0] * iw, xy[1] * ih);
  }

  const eyeWH = (eyeCorner) => {
    const cL = vectorFromPct(eyeCorner[0], iw, ih);
    const cR = vectorFromPct(eyeCorner[2], iw, ih);
    const cT = vectorFromPct(eyeCorner[3], iw, ih);
    const cB = vectorFromPct(eyeCorner[1], iw, ih);
    return [ cL.dist(cR), cT.dist(cB) ]
  }

  const pg = createGraphics(iw, ih);
  pg.clear();
  pg.fill(255);
  pg.noStroke();

  for (let idx = 0; idx < centers.length; idx++) {
    const center = centers[idx];
    const corner = corners[idx];
    if (center[0].length < 1) continue;

    const centerL = vectorFromPct(center[0]);
    const centerR = vectorFromPct(center[1]);
    const centerDiff = p5.Vector.sub(centerR, centerL);

    const centerA = xu.angleBetween(centerDiff);
    const [widthL, heightL] = eyeWH(corner[0]);
    const [widthR, heightR] = eyeWH(corner[1]);
    const centerDist = centerL.dist(centerR);

    pg.push();
    pg.translate(centerL.x, centerL.y);
    pg.rotate(centerA);
    pg.ellipse(0, 0, 2.5*widthL, 2.5*heightL);
    pg.pop();
    
    pg.push();
    pg.translate(centerR.x, centerR.y);
    pg.rotate(centerA);
    pg.ellipse(0, 0, 2.5*widthR, 2.5*heightR);
    pg.pop();
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

  if (styleSel.value() == "ovals") {
    mimg = createEllipsesMask(oimg, cObj["faces"]["mp"]);
  } else {
    mimg = createMasksMask(oimg, cObj["faces"]["mp"]);
  }
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
