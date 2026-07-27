import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

let indexData = null;
let currentSample = null;
let scene, camera, renderer, controls;
let pointCloud = null;
let rgbTexture = null;
let colorMode = "anomaly";

const CATEGORIES_COLORS = {
  bagel: "#22d3ee",
  cable_gland: "#a78bfa",
  carrot: "#fb923c",
  cookie: "#f472b6",
  dowel: "#34d399",
  foam: "#facc15",
};

function $(id) { return document.getElementById(id); }

function lerp(a, b, t) { return a + (b - a) * t; }

function anomalyColor(t) {
  const r = t < 0.5 ? lerp(0.15, 0.95, t * 2) : 0.95;
  const g = t < 0.25 ? lerp(0.25, 0.65, t * 4)
         : t < 0.75 ? lerp(0.65, 0.25, (t - 0.25) * 2)
         : lerp(0.25, 0.05, (t - 0.75) * 4);
  const b = t < 0.5 ? lerp(0.85, 0.25, t * 2) : lerp(0.25, 0.05, (t - 0.5) * 2);
  return [r, g, b];
}

function initScene() {
  const container = $("three-container");
  const w = container.clientWidth;
  const h = container.clientHeight;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0e14);

  camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 10000);
  camera.position.set(0, 0, 200);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(w, h);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.rotateSpeed = 0.6;

  scene.add(new THREE.AmbientLight(0xffffff, 0.4));
  const dir = new THREE.DirectionalLight(0xffffff, 0.6);
  dir.position.set(100, 200, 150);
  scene.add(dir);

  const grid = new THREE.GridHelper(400, 20, 0x1a2030, 0x1a2030);
  grid.position.y = -120;
  scene.add(grid);

  window.addEventListener("resize", onResize);
  animate();
}

function onResize() {
  const c = $("three-container");
  if (!c) return;
  camera.aspect = c.clientWidth / c.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(c.clientWidth, c.clientHeight);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

async function loadIndex() {
  const resp = await fetch("data/index.json");
  indexData = await resp.json();
  $("stats").textContent =
    `${indexData.samples.length} samples · ${indexData.categories.length} categories`;
  populateCategories();
}

function populateCategories() {
  const sel = $("cat-select");
  sel.innerHTML = "";
  for (const cat of indexData.categories) {
    const opt = document.createElement("option");
    opt.value = cat;
    opt.textContent = cat;
    sel.appendChild(opt);
  }
  sel.addEventListener("change", () => renderSampleList(sel.value));
  renderSampleList(sel.value);
}

function renderSampleList(category) {
  const list = $("sample-list");
  list.innerHTML = "";
  const samples = indexData.samples.filter((s) => s.category === category);
  for (const s of samples) {
    const div = document.createElement("div");
    div.className = "sample-item";
    if (currentSample && currentSample.sampleId === s.sampleId && currentSample.category === s.category) {
      div.classList.add("active");
    }
    div.innerHTML =
      `<span class="dot ${s.groundTruth}"></span>` +
      `<span class="meta"><span class="id">${s.sampleId}</span> ` +
      `<span class="dtype">${s.defectType}</span></span>`;
    div.addEventListener("click", () => selectSample(s));
    list.appendChild(div);
  }
  if (samples.length > 0 && !currentSample) {
    selectSample(samples[0]);
  }
}

async function selectSample(sample) {
  currentSample = sample;
  renderSampleList(sample.category);

  const base = `data/${sample.category}/${sample.sampleId}/`;

  $("info-rgb").src = base + "rgb.jpg";
  $("info-heatmap").src = base + "heatmap.png";

  const resp = await fetch(base + "meta.json");
  const meta = await resp.json();

  $("info-score").textContent = meta.calibratedScore.toFixed(4);
  $("info-raw").textContent = meta.rawScore.toFixed(4);

  const gtEl = $("info-gt");
  gtEl.textContent = meta.groundTruth;
  gtEl.style.color = meta.groundTruth === "normal" ? "var(--green)" : "var(--red)";

  $("info-dtype").textContent = meta.defectType;
  $("info-clip").textContent = `"${meta.clipTopPhrase}" (${meta.clipSimilarity.toFixed(4)})`;

  const slider = $("threshold-slider");
  slider.value = meta.calibratedScore;
  $("threshold-val").textContent = meta.calibratedScore.toFixed(3);
  updateBadge(meta.calibratedScore, meta.calibratedScore);

  await loadPointCloud(base + "pointcloud.json", meta);
}

async function loadPointCloud(url, meta) {
  const resp = await fetch(url);
  const pcData = await resp.json();

  if (pointCloud) {
    scene.remove(pointCloud);
    pointCloud.geometry.dispose();
    pointCloud.material.dispose();
    pointCloud = null;
  }

  const positions = new Float32Array(pcData.positions.flat());
  const intensities = new Float32Array(pcData.anomalyIntensity);
  const n = intensities.length;

  const baseColors = new Float32Array(n * 3);
  const rgbColors = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const [r, g, b] = anomalyColor(intensities[i]);
    baseColors[i * 3] = r;
    baseColors[i * 3 + 1] = g;
    baseColors[i * 3 + 2] = b;
    rgbColors[i * 3] = 1;
    rgbColors[i * 3 + 1] = 1;
    rgbColors[i * 3 + 2] = 1;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(baseColors.slice(), 3));
  geometry.computeBoundingSphere();

  const material = new THREE.PointsMaterial({
    size: 1.8,
    vertexColors: true,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.92,
  });

  pointCloud = new THREE.Points(geometry, material);
  pointCloud.userData = { baseColors: baseColors.slice(), rgbColors, intensities, meta };
  scene.add(pointCloud);

  const center = geometry.boundingSphere.center;
  const radius = geometry.boundingSphere.radius;
  camera.position.set(center.x, center.y, center.z + radius * 2.5);
  controls.target.copy(center);
  controls.update();
}

function updateBadge(score, threshold) {
  const badge = $("threshold-badge");
  if (score >= threshold) {
    badge.textContent = "Anomalous";
    badge.className = "badge badge-anomalous";
  } else {
    badge.textContent = "Normal";
    badge.className = "badge badge-normal";
  }
}

function setupEvents() {
  $("enter-btn").addEventListener("click", () => {
    $("landing").classList.add("hidden");
    $("app").classList.remove("hidden");
    initScene();
    onResize();
  });

  $("toggle-color").addEventListener("click", () => {
    colorMode = "anomaly";
    $("toggle-color").classList.add("active");
    $("toggle-color2").classList.remove("active");
    if (pointCloud) {
      pointCloud.geometry.setAttribute(
        "color",
        new THREE.BufferAttribute(pointCloud.userData.baseColors, 3)
      );
      pointCloud.geometry.attributes.color.needsUpdate = true;
    }
  });

  $("toggle-color2").addEventListener("click", () => {
    colorMode = "rgb";
    $("toggle-color2").classList.add("active");
    $("toggle-color").classList.remove("active");
    if (pointCloud) {
      pointCloud.geometry.setAttribute(
        "color",
        new THREE.BufferAttribute(pointCloud.userData.rgbColors, 3)
      );
      pointCloud.geometry.attributes.color.needsUpdate = true;
    }
  });

  $("threshold-slider").addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    $("threshold-val").textContent = val.toFixed(3);
    if (currentSample) {
      const meta = pointCloud ? pointCloud.userData.meta : null;
      const score = meta ? meta.calibratedScore : 0;
      updateBadge(score, val);
    }
  });
}

setupEvents();
loadIndex().catch((err) => {
  console.error("Failed to load index.json:", err);
  $("stats").textContent =
    "Error: could not load data/index.json. Serve this directory with a local server (e.g. python -m http.server 8000)";
});
