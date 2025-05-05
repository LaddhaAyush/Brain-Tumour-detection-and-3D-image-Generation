const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf0f0f0);

// Since you scaled the brain by 3x in Blender, we should adjust the modelScale accordingly
const modelScale = 0.067; // 0.2 / 3 ≈ 0.067
let originalPositions = [];
let explodeFactor = 0;
let brainCenter = new THREE.Vector3();
let tumorBoxCenter = new THREE.Vector3();
let modelLoaded = false;
const clickableMeshes = [];
let brainParts = []; // Store brain parts for organized explosion

// Default tumor coordinates (can be overridden by user input)
let tumorCoordinates = [
  [0.4769948124885559, 0.42294007539749146, 0.4719058833258924],
  [0.5557572245597839, 0.42294007539749146, 0.4719058833258924],
  [0.5557572245597839, 0.4630241394042969, 0.4719058833258924],
  [0.4769948124885559, 0.4630241394042969, 0.4719058833258924],
  [0.4769948124885559, 0.42294007539749146, 0.5280941166741077],
  [0.5557572245597839, 0.42294007539749146, 0.5280941166741077],
  [0.5557572245597839, 0.4630241394042969, 0.5280941166741077],
  [0.4769948124885559, 0.4630241394042969, 0.5280941166741077],
];

// Initialize tumorBoxCenter from default coordinates
tumorCoordinates.forEach((coord) => {
  tumorBoxCenter.x += coord[0];
  tumorBoxCenter.y += coord[1];
  tumorBoxCenter.z += coord[2];
});
tumorBoxCenter.divideScalar(tumorCoordinates.length);

// Track form status
let coordinateFormSubmitted = false;

// Initialize coordinate form elements
let coordinateForm;
let createTumorBtn;
let useDefaultCoordinatesBtn;
let coordinateError;
let changeTumorBtn;

const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
camera.position.set(0, 0, 30);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
document.body.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.minDistance = 0.1;
controls.maxDistance = 50;

// Improved lighting setup with multiple lights
scene.add(new THREE.AmbientLight(0xffffff, 0.5));

// Main directional light (from left)
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(5, 5, 5);
dirLight.castShadow = true;
scene.add(dirLight);

// Add a second directional light from right side
const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.6);
dirLight2.position.set(-5, 3, 4);
dirLight2.castShadow = true;
scene.add(dirLight2);

// Add a fill light from below
const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
fillLight.position.set(0, -5, 0);
scene.add(fillLight);

const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const label = document.getElementById("label") || createLabelElement();
const distanceIndicator =
  document.getElementById("distanceIndicator") || createDistanceIndicator();
const loadingElement =
  document.querySelector(".loading") || createLoadingElement();

// Function to gather coordinates from the form
function getCoordinatesFromForm() {
  const coordinates = [];
  
  for (let i = 1; i <= 8; i++) {
    const x = parseFloat(document.getElementById(`x${i}`).value);
    const y = parseFloat(document.getElementById(`y${i}`).value);
    const z = parseFloat(document.getElementById(`z${i}`).value);
    
    if (isNaN(x) || isNaN(y) || isNaN(z)) {
      return null; // Return null if any coordinate is missing
    }
    
    coordinates.push([x, y, z]);
  }
  
  return coordinates;
}

// Function to validate coordinates
function validateCoordinates(coordinates) {
  if (!coordinates || coordinates.length !== 8) {
    return "Please enter all 8 coordinate points";
  }
  
  // Check if coordinates form a valid bounding box
  // This is a simple check - for a proper implementation, 
  // you might want to check if they form a valid cuboid
  return true;
}

// Function to fill form with default coordinates
function fillDefaultCoordinates() {
  for (let i = 1; i <= 8; i++) {
    document.getElementById(`x${i}`).value = tumorCoordinates[i-1][0];
    document.getElementById(`y${i}`).value = tumorCoordinates[i-1][1];
    document.getElementById(`z${i}`).value = tumorCoordinates[i-1][2];
  }
  coordinateError.textContent = "";
}

// Function to calculate tumor center from coordinates
function calculateTumorCenter(coordinates) {
  const center = new THREE.Vector3();
  
  coordinates.forEach((coord) => {
    center.x += coord[0];
    center.y += coord[1];
    center.z += coord[2];
  });
  
  center.divideScalar(coordinates.length);
  return center;
}

// Function to handle coordinate form submission
function handleCoordinateSubmit() {
  const coordinates = getCoordinatesFromForm();
  const validationResult = validateCoordinates(coordinates);
  
  if (validationResult !== true) {
    coordinateError.textContent = validationResult;
    return;
  }
  
  // Update tumor coordinates
  tumorCoordinates = coordinates;
  tumorBoxCenter = calculateTumorCenter(tumorCoordinates);
  
  // Hide form and show change button
  coordinateForm.style.display = "none";
  changeTumorBtn.style.display = "block";
  coordinateFormSubmitted = true;
  
  // Create/update tumor
  if (modelLoaded) {
    createTumor();
    createTumorAxis();
  }
  
  // Check if we need to load the model
  if (!modelLoaded) {
    loadBrainModel();
  }
}

// Function to show coordinate form again
function showCoordinateForm() {
  coordinateForm.style.display = "block";
}

function createLabelElement() {
  const el = document.createElement("div");
  el.id = "label";
  el.style.display = "none";
  el.style.position = "absolute";
  el.style.background = "rgba(0,0,0,0.7)";
  el.style.color = "white";
  el.style.padding = "5px 10px";
  el.style.borderRadius = "5px";
  el.style.pointerEvents = "none";
  el.style.zIndex = "1001";
  document.body.appendChild(el);
  return el;
}

function createDistanceIndicator() {
  const el = document.createElement("div");
  el.id = "distanceIndicator";
  el.style.position = "absolute";
  el.style.top = "60px";
  el.style.right = "20px";
  el.style.background = "rgba(0,0,0,0.5)";
  el.style.color = "white";
  el.style.padding = "5px 10px";
  el.style.borderRadius = "5px";
  el.style.zIndex = "1001";
  document.body.appendChild(el);
  return el;
}

function createLoadingElement() {
  const el = document.createElement("div");
  el.className = "loading";
  el.style.position = "absolute";
  el.style.top = "50%";
  el.style.left = "50%";
  el.style.transform = "translate(-50%, -50%)";
  el.style.color = "white";
  el.style.background = "rgba(0,0,0,0.7)";
  el.style.padding = "20px";
  el.style.borderRadius = "10px";
  el.style.zIndex = "2000";
  document.body.appendChild(el);
  return el;
}

function createPlaceholderBrain() {
  const brainGroup = new THREE.Group();
  brainGroup.name = "brain_model";

  // Updated with correct names
  const parts = [
    {
      name: "Cerebrum",
      geometry: new THREE.SphereGeometry(0.6, 16, 16),
      position: new THREE.Vector3(0, 0.5, 0.5),
      color: 0x9400d3,
    },
    {
      name: "Corpus",
      geometry: new THREE.SphereGeometry(0.5, 16, 16),
      position: new THREE.Vector3(0, 0.5, -0.3),
      color: 0x800080,
    },
    {
      name: "Frontal Lobe",
      geometry: new THREE.SphereGeometry(0.4, 16, 16),
      position: new THREE.Vector3(0.8, 0, 0.2),
      color: 0x9370db,
    },
    {
      name: "Occupit Lobe",
      geometry: new THREE.SphereGeometry(0.45, 16, 16),
      position: new THREE.Vector3(-0.8, 0, 0.2),
      color: 0x8a2be2,
    },
    {
      name: "Parietal Lobe",
      geometry: new THREE.SphereGeometry(0.4, 16, 16),
      position: new THREE.Vector3(0, 0, -0.8),
      color: 0xd2b48c,
    },
    {
      name: "Pitua Lobe",
      geometry: new THREE.SphereGeometry(0.5, 16, 16),
      position: new THREE.Vector3(0, -0.5, -0.6),
      color: 0xbdb76b,
    },
    {
      name: "Brain Stem",
      geometry: new THREE.CylinderGeometry(0.2, 0.3, 0.8, 16),
      position: new THREE.Vector3(0, -1, -0.3),
      color: 0x006400,
    },
    {
      name: "Temporal Lobe",
      geometry: new THREE.SphereGeometry(0.5, 16, 16),
      position: new THREE.Vector3(0, -0.5, 0.6),
      color: 0x800020,
    },
  ];

  parts.forEach((part) => {
    const material = new THREE.MeshStandardMaterial({
      color: part.color,
      roughness: 0.5,
      metalness: 0.2,
    });

    const mesh = new THREE.Mesh(part.geometry, material);
    mesh.name = part.name;
    mesh.position.copy(part.position);
    originalPositions.push(mesh.position.clone());

    // Add to brainParts array for organized explosion
    brainParts.push({
      mesh: mesh,
      originalPosition: mesh.position.clone(),
      explodeDirection: new THREE.Vector3(
        part.position.x,
        part.position.y,
        part.position.z
      ).normalize(),
    });

    clickableMeshes.push(mesh);
    brainGroup.add(mesh);
  });

  brainGroup.scale.set(modelScale, modelScale, modelScale);
  scene.add(brainGroup);
  brainCenter.set(0, 0, 0);

  createTumor();
  createTumorAxis();

  loadingElement.style.display = "none";
  modelLoaded = true;
  document.body.style.cursor = "auto";
  
  // Show "Change Tumor" button
  changeTumorBtn.style.display = "block";
  
  // Show a success notification
  const notification = document.createElement("div");
  notification.style.position = "absolute";
  notification.style.top = "150px";
  notification.style.left = "50%";
  notification.style.transform = "translateX(-50%)";
  notification.style.background = "rgba(0,128,0,0.8)";
  notification.style.color = "white";
  notification.style.padding = "15px";
  notification.style.borderRadius = "5px";
  notification.style.zIndex = "2000";
  notification.textContent = "Brain model loaded with custom tumor coordinates!";
  document.body.appendChild(notification);

  setTimeout(() => {
    notification.style.opacity = 0;
    notification.style.transition = "opacity 1s";
    setTimeout(() => document.body.removeChild(notification), 1000);
  }, 3000);
}

function loadBrainModel() {
  console.log("Starting to load brain model...");

  // Check if coordinates were submitted
  if (!coordinateFormSubmitted) {
    // User hasn't submitted coordinates yet, let the form handle it
    return;
  }

  let loadingTimeout = setTimeout(() => {
    console.log("Loading timeout reached. Using placeholder brain.");
    createPlaceholderBrain();
  }, 10000);

  loadingElement.style.display = "block";
  loadingElement.textContent = "Loading brain model...";

  const loader = new THREE.GLTFLoader();

  try {
    // Use the original path that was working before
    loader.load(
      "brain.glb",
      function (gltf) {
        clearTimeout(loadingTimeout);
        loadingElement.style.display = "none";
        console.log("Brain model loaded successfully!");

        const model = gltf.scene;
        model.name = "brain_model";
        model.scale.set(modelScale, modelScale, modelScale);

        // Position the model at origin (0,0,0)
        model.position.set(0, 0, 0);

        const box = new THREE.Box3().setFromObject(model);
        box.getCenter(brainCenter);
        console.log("Brain center calculated at:", brainCenter);

        brainParts = []; // Clear existing parts
        originalPositions = [];
        clickableMeshes.length = 0;

        // Array of colors matching your second image
        const brainColors = [
          0x9400d3, // Purple
          0x800080, // Dark purple
          0x9370db, // Medium purple
          0x8a2be2, // Blue violet
          0xd2b48c, // Tan
          0xbdb76b, // Dark khaki
          0x006400, // Dark green
          0x800020, // Burgundy
        ];

        // Predefined part names for the brain model
        const partNames = [
          "Cerebrum",
          "Corpus",
          "Frontal Lobe",
          "Occupit Lobe",
          "Parietal Lobe",
          "Pitua Lobe",
          "Brain Stem",
          "Temporal Lobe",
        ];

        // Get all meshes from the model
        const meshes = [];
        model.traverse((child) => {
          if (child.isMesh) {
            meshes.push(child);
          }
        });

        // Process up to 8 meshes and assign predefined colors
        meshes.slice(0, 8).forEach((child, index) => {
          // Store original position
          const origPos = child.position.clone();
          originalPositions.push(origPos);

          // Assign name and color
          const partName = partNames[index % partNames.length];
          const color = brainColors[index % brainColors.length];
          child.name = partName;

          // Set up materials
          child.castShadow = true;
          child.receiveShadow = true;
          child.material = new THREE.MeshStandardMaterial({
            color: color,
            roughness: 0.5,
            metalness: 0.2,
            transparent: true,
          });

          // Calculate explosion direction based on position relative to tumor
          const direction = new THREE.Vector3();
          direction.subVectors(origPos, brainCenter).normalize();

          // Add to brain parts for organized explosion
          brainParts.push({
            mesh: child,
            originalPosition: origPos.clone(),
            explodeDirection: direction,
          });

          clickableMeshes.push(child);
        });

        scene.add(model);
        createTumor();
        createTumorAxis();

        // Set camera to look at brain center
        camera.lookAt(brainCenter);
        controls.target.copy(brainCenter);

        modelLoaded = true;
        document.body.style.cursor = "auto";

        // Success notification
        const notification = document.createElement("div");
        notification.style.position = "absolute";
        notification.style.top = "150px";
        notification.style.left = "50%";
        notification.style.transform = "translateX(-50%)";
        notification.style.background = "rgba(0,128,0,0.8)";
        notification.style.color = "white";
        notification.style.padding = "15px";
        notification.style.borderRadius = "5px";
        notification.style.zIndex = "2000";
        notification.textContent = "Enhanced brain model loaded successfully!";
        document.body.appendChild(notification);

        setTimeout(() => {
          notification.style.opacity = 0;
          notification.style.transition = "opacity 1s";
          setTimeout(() => document.body.removeChild(notification), 1000);
        }, 3000);
      },
      function (xhr) {
        if (xhr.total > 0) {
          const percent = ((xhr.loaded / xhr.total) * 100).toFixed(0);
          loadingElement.textContent = `Loading brain model... ${percent}%`;
          console.log(`Loading progress: ${percent}%`);
        }
      },
      function (error) {
        clearTimeout(loadingTimeout);
        console.error("Error loading brain model:", error);
        createPlaceholderBrain();
      }
    );
  } catch (error) {
    clearTimeout(loadingTimeout);
    console.error("Exception when loading brain model:", error);
    createPlaceholderBrain();
  }
}

function createTumor() {
  // Remove existing tumor objects if present
  const existingTumor = scene.getObjectByName("tumor");
  if (existingTumor) scene.remove(existingTumor);

  const existingTumorBox = scene.getObjectByName("tumor_box");
  if (existingTumorBox) scene.remove(existingTumorBox);

  // Scale tumor center relative to model scale
  const scaledTumorCenter = tumorBoxCenter.clone().multiplyScalar(modelScale);

  // Create a more realistic tumor with appropriate size for 3x scaled model
  const tumorGeometry = new THREE.SphereGeometry(0.15 * modelScale, 32, 32);
  const tumorMaterial = new THREE.MeshStandardMaterial({
    color: 0xff0000,
    emissive: 0xff0000,
    emissiveIntensity: 0.5,
    roughness: 0.3,
    metalness: 0.5,
  });

  const tumor = new THREE.Mesh(tumorGeometry, tumorMaterial);
  tumor.name = "tumor";
  tumor.position.copy(scaledTumorCenter);
  scene.add(tumor);

  // Add pulsing animation to the tumor
  gsap.to(tumor.scale, {
    x: 1.3,
    y: 1.3,
    z: 1.3,
    duration: 0.8,
    yoyo: true,
    repeat: -1,
  });

  // Calculate box dimensions from user-provided coordinates
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  let minZ = Infinity, maxZ = -Infinity;
  
  // Find the min and max values to create the bounding box
  tumorCoordinates.forEach(coord => {
    minX = Math.min(minX, coord[0]);
    maxX = Math.max(maxX, coord[0]);
    minY = Math.min(minY, coord[1]);
    maxY = Math.max(maxY, coord[1]);
    minZ = Math.min(minZ, coord[2]);
    maxZ = Math.max(maxZ, coord[2]);
  });
  
  const boxWidth = (maxX - minX) * modelScale;
  const boxHeight = (maxY - minY) * modelScale;
  const boxDepth = (maxZ - minZ) * modelScale;

  const boxGeometry = new THREE.BoxGeometry(
    boxWidth * 3,
    boxHeight * 3,
    boxDepth * 3
  );
  const boxMaterial = new THREE.MeshBasicMaterial({
    color: 0xff0000,
    wireframe: true,
    transparent: true,
    opacity: 0.7,
  });

  const tumorBox = new THREE.Mesh(boxGeometry, boxMaterial);
  tumorBox.name = "tumor_box";
  tumorBox.position.copy(scaledTumorCenter);
  scene.add(tumorBox);
  
  // Display the tumor coordinates in the console for debugging
  console.log("Tumor created at coordinates:", tumorCoordinates);
  console.log("Tumor center:", tumorBoxCenter);
}

function createTumorAxis() {
  // Remove existing axis elements
  const existingAxis = scene.getObjectByName("tumor_axis");
  if (existingAxis) scene.remove(existingAxis);
  
  ["x_label", "y_label", "z_label"].forEach(name => {
    const existingLabel = scene.getObjectByName(name);
    if (existingLabel) scene.remove(existingLabel);
  });

  const scaledTumorCenter = tumorBoxCenter.clone().multiplyScalar(modelScale);

  const axisLength = 0.5;
  const axesHelper = new THREE.AxesHelper(axisLength);
  axesHelper.position.copy(scaledTumorCenter);
  axesHelper.name = "tumor_axis";
  axesHelper.visible = false;
  scene.add(axesHelper);

  const createLabel = (text, position, color) => {
    const canvas = document.createElement("canvas");
    canvas.width = 64;
    canvas.height = 32;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = color;
    ctx.font = "24px Arial";
    ctx.fillText(text, 4, 24);

    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({ map: texture });
    const sprite = new THREE.Sprite(material);
    sprite.position.copy(position);
    sprite.scale.set(0.2, 0.1, 1);
    return sprite;
  };

  const xPos = scaledTumorCenter
    .clone()
    .add(new THREE.Vector3(axisLength * 1.2, 0, 0));
  const yPos = scaledTumorCenter
    .clone()
    .add(new THREE.Vector3(0, axisLength * 1.2, 0));
  const zPos = scaledTumorCenter
    .clone()
    .add(new THREE.Vector3(0, 0, axisLength * 1.2));

  const xLabel = createLabel("X", xPos, "#ff0000");
  const yLabel = createLabel("Y", yPos, "#00ff00");
  const zLabel = createLabel("Z", zPos, "#0000ff");

  xLabel.name = "x_label";
  yLabel.name = "y_label";
  zLabel.name = "z_label";

  xLabel.visible = false;
  yLabel.visible = false;
  zLabel.visible = false;

  scene.add(xLabel);
  scene.add(yLabel);
  scene.add(zLabel);
}

function toggleTumorAxis() {
  const axis = scene.getObjectByName("tumor_axis");
  const xLabel = scene.getObjectByName("x_label");
  const yLabel = scene.getObjectByName("y_label");
  const zLabel = scene.getObjectByName("z_label");

  if (axis) {
    axis.visible = !axis.visible;
    xLabel.visible = axis.visible;
    yLabel.visible = axis.visible;
    zLabel.visible = axis.visible;

    // Remove existing tumor coordinates display
    const existingCoords = document.getElementById("tumor-coords");
    if (existingCoords) existingCoords.remove();

    if (axis.visible) {
      const tumorCoords = document.createElement("div");
      tumorCoords.id = "tumor-coords";
      tumorCoords.style.position = "absolute";
      tumorCoords.style.bottom = "20px";
      tumorCoords.style.right = "20px";
      tumorCoords.style.background = "rgba(0,0,0,0.7)";
      tumorCoords.style.color = "white";
      tumorCoords.style.padding = "10px";
      tumorCoords.style.borderRadius = "5px";
      tumorCoords.style.fontFamily = "monospace";

      // Display tumor center coordinates
      const coords = tumorBoxCenter.clone();
      let html = `<strong>Tumor Center:</strong><br>
                X: ${coords.x.toFixed(3)}<br>
                Y: ${coords.y.toFixed(3)}<br>
                Z: ${coords.z.toFixed(3)}<br><br>
                <strong>All Points:</strong><br>`;

      // Display all 8 coordinates
      tumorCoordinates.forEach((point, index) => {
        html += `Point ${index + 1}: (${point[0].toFixed(3)}, ${point[1].toFixed(3)}, ${point[2].toFixed(3)})<br>`;
      });

      tumorCoords.innerHTML = html;
      document.body.appendChild(tumorCoords);
    }
  }
}

function cycleSelectedPart() {
  if (!modelLoaded) return;

  let currentIndex = -1;

  clickableMeshes.forEach((mesh, index) => {
    if (mesh.userData.selected) {
      currentIndex = index;
      mesh.userData.selected = false;
      if (mesh.material) mesh.material.emissive = new THREE.Color(0x000000);
    }
  });

  const nextIndex = (currentIndex + 1) % clickableMeshes.length;
  const nextPart = clickableMeshes[nextIndex];

  if (nextPart && nextPart.name !== "tumor" && nextPart.name !== "tumor_box") {
    nextPart.userData.selected = true;
    if (nextPart.material)
      nextPart.material.emissive = new THREE.Color(0xffff00);

    let selectedLabel = document.getElementById("selected-part-label");
    if (!selectedLabel) {
      selectedLabel = document.createElement("div");
      selectedLabel.id = "selected-part-label";
      selectedLabel.style.position = "absolute";
      selectedLabel.style.background = "rgba(0,0,0,0.7)";
      selectedLabel.style.color = "white";
      selectedLabel.style.padding = "8px 12px";
      selectedLabel.style.borderRadius = "5px";
      selectedLabel.style.fontSize = "14px";
      selectedLabel.style.fontWeight = "bold";
      selectedLabel.style.top = "100px";
      selectedLabel.style.left = "50%";
      selectedLabel.style.transform = "translateX(-50%)";
      selectedLabel.style.zIndex = "1000";
      document.body.appendChild(selectedLabel);
    }

    const partName = nextPart.name || "Brain Part " + (nextIndex + 1);
    selectedLabel.textContent = `Selected: ${partName}`;

    // First zoom out a bit to avoid getting lost in white space
    gsap.to(camera.position, {
      x: camera.position.x * 1.5,
      y: camera.position.y * 1.5,
      z: camera.position.z * 1.5,
      duration: 0.4,
      ease: "power1.out",
      onComplete: () => {
        // Then move to the part's position
        gsap.to(controls.target, {
          x: nextPart.position.x,
          y: nextPart.position.y,
          z: nextPart.position.z,
          duration: 0.8,
          ease: "power2.inOut",
        });
      },
    });
  }
}

// Improved animate function with better explosion animation for 8 distinct brain parts
// Complete updated animate function
function animate() {
  requestAnimationFrame(animate);
  controls.update();

  if (modelLoaded) {
    const distanceToBrain = camera.position.distanceTo(brainCenter);
    distanceIndicator.textContent = `Distance: ${distanceToBrain.toFixed(2)}`;

    if (window.forceExplode !== true && window.forceExplode !== false) {
      if (
        distanceToBrain < 8 &&
        explodeFactor < 0.5 &&
        !window.animatingExplosion
      ) {
        window.animatingExplosion = true;
        gsap.to(window, {
          explodeFactor: 1,
          duration: 1.2,
          ease: "power2.out",
          onComplete: () => {
            window.animatingExplosion = false;
          },
        });

        const explodeButton = document.querySelector(".explode-btn");
        if (explodeButton) {
          explodeButton.textContent = "REFORM BRAIN";
          explodeButton.style.background = "#33aa33";
        }
      } else if (
        distanceToBrain > 18 &&
        explodeFactor > 0.5 &&
        !window.animatingExplosion
      ) {
        window.animatingExplosion = true;
        gsap.to(window, {
          explodeFactor: 0,
          duration: 1.5,
          ease: "power2.inOut",
          onComplete: () => {
            window.animatingExplosion = false;
          },
        });

        const explodeButton = document.querySelector(".explode-btn");
        if (explodeButton) {
          explodeButton.textContent = "EXPLODE BRAIN";
          explodeButton.style.background = "#ff3333";
        }
      }
    }

    explodeFactor = window.explodeFactor || 0;

    const tumor = scene.getObjectByName("tumor");
    const tumorPos = tumor ? tumor.position.clone() : brainCenter.clone();

    // Use the brainParts array for more organized explosion in 8 directions
    if (brainParts.length > 0) {
      brainParts.forEach((part, index) => {
        if (
          !part.mesh ||
          part.mesh.name === "tumor" ||
          part.mesh.name === "tumor_box"
        )
          return;

        // Custom direction based on part name for better separation
        let explodeDirection = new THREE.Vector3();
        const partName = part.mesh.name;
        
        // Assign specific directions based on part names to avoid overlapping
        if (partName === "Cerebrum") {
          explodeDirection.set(0, 15, 5); // Top center
        } else if (partName === "Corpus") {
          explodeDirection.set(0, 12, -5); // Top back
        } else if (partName === "Frontal Lobe") {
          explodeDirection.set(15, 5, 0); // Far right
        } else if (partName === "Occupit Lobe") {
          explodeDirection.set(-15, 5, 0); // Far left
        } else if (partName === "Parietal Lobe") {
          explodeDirection.set(0, -5, -15); // Bottom back
        } else if (partName === "Pitua Lobe") {
          explodeDirection.set(12, -8, 5); // Bottom right
        } else if (partName === "Temporal Lobe") {
          explodeDirection.set(-12, -8, 5); // Bottom left
        } else if (partName === "Brain Stem") {
          explodeDirection.set(0, -15, 0); // Straight down
        } else {
          // Fallback to using the original direction
          explodeDirection = part.explodeDirection.clone();
        }
        
        // Normalize and apply explosion distance
        explodeDirection.normalize();
        const explosionDistance = 375 * modelScale; // 2.5x more explosion
        explodeDirection.multiplyScalar(explosionDistance);

        // Apply easing to the explosion effect
        const easedFactor = explodeFactor * (2 - explodeFactor);
        const newPos = new THREE.Vector3();
        newPos.copy(part.originalPosition);
        newPos.add(explodeDirection.multiplyScalar(easedFactor));

        part.mesh.position.copy(newPos);

        if (part.mesh.material) {
          part.mesh.material.transparent = true;
          part.mesh.material.opacity = 1 - explodeFactor * 0.4;
        }
      });
    } else {
      // Fallback to original code if brainParts isn't populated
      clickableMeshes.forEach((part, index) => {
        if (part.name === "tumor" || part.name === "tumor_box") return;

        const originalPos = originalPositions[index] || part.position.clone();
        let explodeDirection = new THREE.Vector3();
        
        // Matching directions with the code above
        const partName = part.name;
        if (partName === "Cerebrum") {
          explodeDirection.set(0, 15, 5); // Top center
        } else if (partName === "Corpus") {
          explodeDirection.set(0, 12, -5); // Top back
        } else if (partName === "Frontal Lobe") {
          explodeDirection.set(15, 5, 0); // Far right
        } else if (partName === "Occupit Lobe") {
          explodeDirection.set(-15, 5, 0); // Far left
        } else if (partName === "Parietal Lobe") {
          explodeDirection.set(0, -5, -15); // Bottom back
        } else if (partName === "Pitua Lobe") {
          explodeDirection.set(12, -8, 5); // Bottom right
        } else if (partName === "Temporal Lobe") {
          explodeDirection.set(-12, -8, 5); // Bottom left
        } else if (partName === "Brain Stem") {
          explodeDirection.set(0, -15, 0); // Straight down
        } else {
          // Default directions based on index if name doesn't match
          switch (index % 8) {
            case 0: explodeDirection.set(10, 10, 10); break;
            case 1: explodeDirection.set(-10, 10, 10); break;
            case 2: explodeDirection.set(10, 10, -10); break;
            case 3: explodeDirection.set(-10, 10, -10); break;
            case 4: explodeDirection.set(10, -10, 10); break;
            case 5: explodeDirection.set(-10, -10, 10); break;
            case 6: explodeDirection.set(10, -10, -10); break;
            case 7: explodeDirection.set(-10, -10, -10); break;
            default:
              explodeDirection = originalPos.clone().sub(tumorPos).normalize().multiplyScalar(20);
          }
        }

        const explosionDistance = 375 * modelScale; // 2.5x more explosion
        explodeDirection.normalize().multiplyScalar(explosionDistance);

        const easedFactor = explodeFactor * (2 - explodeFactor);
        const newPos = new THREE.Vector3();
        newPos.copy(originalPos);
        newPos.add(explodeDirection.multiplyScalar(easedFactor));

        part.position.copy(newPos);

        if (part.material) {
          part.material.transparent = true;
          part.material.opacity = 1 - explodeFactor * 0.4;
        }
      });
    }

    // Tumor label display logic
    if (tumor && explodeFactor > 0.5 && !window.tumorLabelCreated) {
      const tumorLabel = document.createElement("div");
      tumorLabel.id = "tumor-label";
      tumorLabel.textContent = "TUMOR";
      tumorLabel.style.position = "absolute";
      tumorLabel.style.background = "rgba(255,0,0,0.8)";
      tumorLabel.style.color = "white";
      tumorLabel.style.padding = "5px 10px";
      tumorLabel.style.borderRadius = "5px";
      tumorLabel.style.fontWeight = "bold";
      tumorLabel.style.fontSize = "14px";
      tumorLabel.style.transform = "translate(-50%, -50%)";
      tumorLabel.style.pointerEvents = "none";
      document.body.appendChild(tumorLabel);
      window.tumorLabelCreated = true;
    } else if (explodeFactor < 0.5 && window.tumorLabelCreated) {
      const tumorLabel = document.getElementById("tumor-label");
      if (tumorLabel) {
        tumorLabel.remove();
        window.tumorLabelCreated = false;
      }
    }

    const tumorLabel = document.getElementById("tumor-label");
    if (tumor && tumorLabel) {
      const screenPosition = new THREE.Vector3();
      screenPosition.copy(tumor.position);
      screenPosition.project(camera);

      const x = (screenPosition.x * 0.5 + 0.5) * window.innerWidth;
      const y = (-(screenPosition.y * 0.5) + 0.5) * window.innerHeight;

      tumorLabel.style.left = x + "px";
      tumorLabel.style.top = y + "px";
    }
  }

  renderer.render(scene, camera);
}

function setupEventHandlers() {
  window.addEventListener("click", (event) => {
    if (!modelLoaded) return;

    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(clickableMeshes, true);

    if (intersects.length > 0) {
      const part = intersects[0].object;

      label.style.display = "block";
      label.style.left = `${event.clientX + 10}px`;
      label.style.top = `${event.clientY}px`;

      const partName = part.name || "Brain Part";
      label.textContent = partName;

      if (part.material) {
        gsap.to(part.material, {
          emissive: new THREE.Color(0xffff00),
          duration: 0.3,
        });
        setTimeout(() => {
          gsap.to(part.material, {
            emissive: new THREE.Color(0x000000),
            duration: 0.5,
          });
        }, 1000);
      }

      setTimeout(() => {
        label.style.display = "none";
      }, 3000);
    }
  });

  window.addEventListener("mousemove", (event) => {
    if (!modelLoaded) return;

    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(clickableMeshes, true);

    clickableMeshes.forEach((mesh) => {
      if (mesh.material && mesh.userData.hovered && !mesh.userData.selected) {
        mesh.material.emissive = new THREE.Color(0x000000);
        mesh.userData.hovered = false;
      }
    });

    if (intersects.length > 0) {
      const part = intersects[0].object;
      if (part.material && !part.userData.selected) {
        part.material.emissive = new THREE.Color(0x333333);
        part.userData.hovered = true;

        const partName = part.name || "Brain Part";
        label.textContent = partName;
        label.style.display = "block";
        label.style.left = `${event.clientX + 10}px`;
        label.style.top = `${event.clientY}px`;
      }
    } else {
      if (!document.getElementById("selected-part-label")) {
        label.style.display = "none";
      }
    }
  });

  window.addEventListener("dblclick", () => {
    if (!modelLoaded) return;

    const tumor = scene.getObjectByName("tumor");
    if (tumor) {
      gsap.to(controls.target, {
        x: tumor.position.x,
        y: tumor.position.y,
        z: tumor.position.z,
        duration: 1.2,
        ease: "power2.inOut",
      });

      const direction = camera.position.clone().sub(tumor.position).normalize();
      const targetPosition = tumor.position
        .clone()
        .add(direction.multiplyScalar(5));

      gsap.to(camera.position, {
        x: targetPosition.x,
        y: targetPosition.y,
        z: targetPosition.z,
        duration: 1.2,
        ease: "power2.inOut",
        onComplete: () => {
          window.forceExplode = true;

          window.animatingExplosion = true;
          gsap.to(window, {
            explodeFactor: 1,
            duration: 1.2,
            onComplete: () => {
              window.animatingExplosion = false;

              const axis = scene.getObjectByName("tumor_axis");
              if (axis) {
                axis.visible = true;
                scene.getObjectByName("x_label").visible = true;
                scene.getObjectByName("y_label").visible = true;
                scene.getObjectByName("z_label").visible = true;

                const tumorCoords = document.createElement("div");
                tumorCoords.id = "tumor-coords";
                tumorCoords.style.position = "absolute";
                tumorCoords.style.bottom = "20px";
                tumorCoords.style.right = "20px";
                tumorCoords.style.background = "rgba(0,0,0,0.7)";
                tumorCoords.style.color = "white";
                tumorCoords.style.padding = "10px";
                tumorCoords.style.borderRadius = "5px";
                tumorCoords.style.fontFamily = "monospace";

                const coords = tumorBoxCenter.clone();
                tumorCoords.innerHTML = `Tumor Coordinates:<br>
                                         X: ${coords.x.toFixed(3)}<br>
                                         Y: ${coords.y.toFixed(3)}<br>
                                         Z: ${coords.z.toFixed(3)}`;
                document.body.appendChild(tumorCoords);
              }
            },
          });

          const explodeButton = document.querySelector(".explode-btn");
          if (explodeButton) {
            explodeButton.textContent = "REFORM BRAIN";
            explodeButton.style.background = "#33aa33";
          }
        },
      });
    }
  });

  window.addEventListener("keydown", (event) => {
    if (!modelLoaded) return;

    switch (event.key) {
      case "+":
      case "=":
        adjustModelScale(1.2);
        break;
      case "-":
      case "_":
        adjustModelScale(0.8);
        break;
      case "r":
      case "R":
        resetView();
        break;
      case " ":
        event.preventDefault();
        toggleExplode();
        break;
      case "a":
      case "A":
        toggleTumorAxis();
        break;
      case "Tab":
        event.preventDefault();
        cycleSelectedPart();
        break;
    }
  });

  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });
}

function toggleExplode() {
  window.forceExplode = !window.forceExplode;

  if (window.forceExplode) {
    window.animatingExplosion = true;
    gsap.to(window, {
      explodeFactor: 1,
      duration: 1.2,
      ease: "power2.out",
      onComplete: () => {
        window.animatingExplosion = false;
      },
    });
  } else {
    window.animatingExplosion = true;
    gsap.to(window, {
      explodeFactor: 0,
      duration: 1.5,
      ease: "power2.inOut",
      onComplete: () => {
        window.animatingExplosion = false;
      },
    });
  }

  const explodeButton = document.querySelector(".explode-btn");
  if (explodeButton) {
    explodeButton.textContent = window.forceExplode
      ? "REFORM BRAIN"
      : "EXPLODE BRAIN";
    explodeButton.style.background = window.forceExplode
      ? "#33aa33"
      : "#ff3333";
  }
}

function resetView() {
  gsap.to(camera.position, {
    x: 0,
    y: 0,
    z: 30,
    duration: 1.5,
    ease: "power2.inOut",
  });

  gsap.to(controls.target, {
    x: 0,
    y: 0,
    z: 0,
    duration: 1.5,
    ease: "power2.inOut",
  });

  window.forceExplode = false;

  window.animatingExplosion = true;
  gsap.to(window, {
    explodeFactor: 0,
    duration: 1.5,
    ease: "power2.inOut",
    onComplete: () => {
      window.animatingExplosion = false;
    },
  });

  const axis = scene.getObjectByName("tumor_axis");
  if (axis) {
    axis.visible = false;
    scene.getObjectByName("x_label").visible = false;
    scene.getObjectByName("y_label").visible = false;
    scene.getObjectByName("z_label").visible = false;

    const tumorCoords = document.getElementById("tumor-coords");
    if (tumorCoords) tumorCoords.remove();
  }

  clickableMeshes.forEach((mesh) => {
    if (mesh.userData.selected) {
      mesh.userData.selected = false;
      if (mesh.material) mesh.material.emissive = new THREE.Color(0x000000);
    }
  });

  const selectedLabel = document.getElementById("selected-part-label");
  if (selectedLabel) selectedLabel.remove();

  const explodeButton = document.querySelector(".explode-btn");
  if (explodeButton) {
    explodeButton.textContent = "EXPLODE BRAIN";
    explodeButton.style.background = "#ff3333";
  }
}

function adjustModelScale(factor) {
  if (!modelLoaded) return;

  const model = scene.getObjectByName("brain_model");
  if (model) {
    model.scale.multiplyScalar(factor);

    const tumor = scene.getObjectByName("tumor");
    if (tumor) {
      tumor.position.multiplyScalar(factor);
      tumor.scale.multiplyScalar(factor);
    }

    const tumorBox = scene.getObjectByName("tumor_box");
    if (tumorBox) {
      tumorBox.position.multiplyScalar(factor);
      tumorBox.scale.multiplyScalar(factor);
    }

    const tumorAxis = scene.getObjectByName("tumor_axis");
    if (tumorAxis) {
      tumorAxis.position.multiplyScalar(factor);
      tumorAxis.scale.multiplyScalar(factor);
    }

    ["x_label", "y_label", "z_label"].forEach((labelName) => {
      const label = scene.getObjectByName(labelName);
      if (label) label.position.multiplyScalar(factor);
    });

    controls.minDistance *= factor;
    controls.maxDistance *= factor;
  }
}

// Function to adjust material properties
function adjustModelMaterials() {
  const model = scene.getObjectByName("brain_model");
  if (!model) return;

  model.traverse((child) => {
    if (child.isMesh) {
      // Enhance material properties for better visualization
      if (child.material) {
        child.material.roughness = 0.7;
        child.material.metalness = 0.2;
        child.material.needsUpdate = true;
      }
    }
  });

  const notification = document.createElement("div");
  notification.style.position = "absolute";
  notification.style.top = "150px";
  notification.style.left = "50%";
  notification.style.transform = "translateX(-50%)";
  notification.style.background = "rgba(0,128,0,0.8)";
  notification.style.color = "white";
  notification.style.padding = "15px";
  notification.style.borderRadius = "5px";
  notification.style.zIndex = "2000";
  notification.textContent =
    "Material properties adjusted for better visualization";
  document.body.appendChild(notification);

  setTimeout(() => {
    notification.style.opacity = 0;
    notification.style.transition = "opacity 1s";
    setTimeout(() => document.body.removeChild(notification), 1000);
  }, 2000);
}

document.addEventListener("DOMContentLoaded", () => {
  // Initialize coordinate form elements
  coordinateForm = document.getElementById("tumorCoordinateForm");
  createTumorBtn = document.getElementById("createTumorBtn");
  useDefaultCoordinatesBtn = document.getElementById("useDefaultCoordinatesBtn");
  coordinateError = document.getElementById("coordinateError");
  changeTumorBtn = document.getElementById("change-tumor-btn");
  
  // Set up event listeners for coordinate form
  createTumorBtn.addEventListener("click", handleCoordinateSubmit);
  useDefaultCoordinatesBtn.addEventListener("click", () => {
    fillDefaultCoordinates();
    handleCoordinateSubmit();
  });
  
  // Connect change tumor button if using the old UI
  if (changeTumorBtn) {
  changeTumorBtn.addEventListener("click", showCoordinateForm);
  }
  
  // UI elements for brain controls now in HTML

  // Start with coordinate form visible, brain will load after form submission
  coordinateForm.style.display = "block";
  if (changeTumorBtn) changeTumorBtn.style.display = "none";
  
  // Initialize the 3D environment and event handlers
  setupEventHandlers();
  animate();
});
