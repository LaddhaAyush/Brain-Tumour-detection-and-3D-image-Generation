const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1b2a); // Changed from 0xf0f0f0 to dark navy

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

// Brain part coordinates from blender model
const brainPartCoordinates = {
  pituitary: { x: 8.0956, y: 0.4992, z: -6.8596 },
  frontal: { x: 11.812, y: -0.30038, z: 8.742 },
  parietal: { x: -7.2249, y: -0.30014, z: 10.883 },
  occipital: { x: -15.42, y: -0.30001, z: -1.0365 },
  cerebellum: { x: -7.0888, y: 0.49958, z: -9.1671 },
  brainStem: { x: 3.9349, y: 0.49944, z: -5.6843 },
  corpus: { x: 3.4753, y: -0.30032, z: 5.9531 },
  temporal: { x: 2.3569, y: -0.30022, z: -2.3855 }
};

// Function to generate anatomically accurate coordinates for tumor types
function generateTumorCoordinatesForType(tumorType) {
  console.log("Generating coordinates for tumor type:", tumorType);
  
  // Base coordinates for the tumor's center
  let baseCoords;
  const size = 0.2; // Size of tumor "box"
  
  if (tumorType && tumorType.toLowerCase().includes('pituitary')) {
    // Use exact pituitary coordinates
    baseCoords = {
      x: brainPartCoordinates.pituitary.x,
      y: brainPartCoordinates.pituitary.y,
      z: brainPartCoordinates.pituitary.z
    };
    console.log("Using pituitary coordinates for tumor:", baseCoords);
    
    // Make pituitary coordinates more visible by adjusting them slightly
    // This is a temporary workaround to ensure pituitary tumors are properly visible
    baseCoords.x = 0; // Center horizontally
    baseCoords.y = -0.6; // Position below
    baseCoords.z = 0.6; // Position forward
    console.log("Adjusted pituitary coordinates for better visibility:", baseCoords);
  } 
  else if (tumorType && tumorType.toLowerCase().includes('glioma')) {
    // For gliomas, use coordinates from cerebral hemispheres (frontal, parietal, or temporal)
    const gliomaParts = ['frontal', 'parietal', 'temporal'];
    const selectedPart = gliomaParts[Math.floor(Math.random() * gliomaParts.length)];
    baseCoords = {
      x: brainPartCoordinates[selectedPart].x,
      y: brainPartCoordinates[selectedPart].y,
      z: brainPartCoordinates[selectedPart].z
    };
    console.log(`Using ${selectedPart} coordinates for glioma tumor`);
  }
  else if (tumorType && tumorType.toLowerCase().includes('meningioma')) {
    // For meningiomas, position near the surface between parts
    baseCoords = {
      x: (brainPartCoordinates.frontal.x + brainPartCoordinates.parietal.x) / 2,
      y: brainPartCoordinates.frontal.y - 0.5,
      z: (brainPartCoordinates.frontal.z + brainPartCoordinates.parietal.z) / 2
    };
    console.log("Using surface coordinates for meningioma tumor");
  }
  else {
    // Default (unknown type) - use pituitary as fallback
    baseCoords = {
      x: brainPartCoordinates.pituitary.x,
      y: brainPartCoordinates.pituitary.y,
      z: brainPartCoordinates.pituitary.z
    };
    console.log("Using default/pituitary coordinates for unknown tumor type");
  }
  
  // Generate 8 vertices of a tumor box centered at baseCoords
  return [
    [baseCoords.x - size, baseCoords.y - size, baseCoords.z - size],
    [baseCoords.x + size, baseCoords.y - size, baseCoords.z - size],
    [baseCoords.x + size, baseCoords.y + size, baseCoords.z - size],
    [baseCoords.x - size, baseCoords.y + size, baseCoords.z - size],
    [baseCoords.x - size, baseCoords.y - size, baseCoords.z + size],
    [baseCoords.x + size, baseCoords.y - size, baseCoords.z + size],
    [baseCoords.x + size, baseCoords.y + size, baseCoords.z + size],
    [baseCoords.x - size, baseCoords.y + size, baseCoords.z + size]
  ];
}

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
  // Get the tumor type from URL params if available
  const urlParams = new URLSearchParams(window.location.search);
  const tumorType = urlParams.get('tumor_type') || 'pituitary'; // Default to pituitary if not specified
  
  // Generate anatomically accurate coordinates based on tumor type
  const anatomicalCoordinates = generateTumorCoordinatesForType(tumorType);
  
  // Use these coordinates to fill the form
  for (let i = 1; i <= 8; i++) {
    document.getElementById(`x${i}`).value = anatomicalCoordinates[i-1][0];
    document.getElementById(`y${i}`).value = anatomicalCoordinates[i-1][1];
    document.getElementById(`z${i}`).value = anatomicalCoordinates[i-1][2];
  }
  
  // Update global tumor coordinates
  tumorCoordinates = anatomicalCoordinates;
  tumorBoxCenter = calculateTumorCenter(tumorCoordinates);
  
  console.log("Updated tumor coordinates for type:", tumorType);
  console.log("New coordinates:", tumorCoordinates);
  console.log("New center:", tumorBoxCenter);
  
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
  el.style.top = "20px";
  el.style.right = "20px";
  el.style.background = "rgba(40, 40, 40, 0.85)";
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
    
    // Also update the loading screen when using placeholder
    if (typeof updateLoadingProgress === 'function') {
      updateLoadingProgress(100);
    } else {
      // If the function doesn't exist, try to manipulate elements directly
      const loadingScreen = document.getElementById('loadingScreen');
      const loadingBar = document.getElementById('loadingBar');
      const loadingPercent = document.getElementById('loadingPercent');
      
      if (loadingBar) loadingBar.style.width = '100%';
      if (loadingPercent) loadingPercent.textContent = '100%';
      
      if (loadingScreen) {
        setTimeout(() => {
          loadingScreen.style.opacity = '0';
          setTimeout(() => {
            loadingScreen.style.display = 'none';
          }, 500);
        }, 300);
      }
    }
  }, 10000);

  loadingElement.style.display = "block";
  loadingElement.textContent = "Loading brain model...";

  const loader = new THREE.GLTFLoader();

  try {
    // Use the updated path to work with Flask's static folder
    loader.load(
      "/static/brain.glb",
      function (gltf) {
        clearTimeout(loadingTimeout);
        loadingElement.style.display = "none";
        console.log("Brain model loaded successfully!");

        // Update the loading progress to 100% when loading completes
        if (typeof updateLoadingProgress === 'function') {
          updateLoadingProgress(100);
        } else {
          // If the function doesn't exist, try to manipulate elements directly
          const loadingScreen = document.getElementById('loadingScreen');
          const loadingBar = document.getElementById('loadingBar');
          const loadingPercent = document.getElementById('loadingPercent');
          
          if (loadingBar) loadingBar.style.width = '100%';
          if (loadingPercent) loadingPercent.textContent = '100%';
          
          if (loadingScreen) {
            setTimeout(() => {
              loadingScreen.style.opacity = '0';
              setTimeout(() => {
                loadingScreen.style.display = 'none';
              }, 500);
            }, 300);
          }
        }

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
          0x05c2c9, // Primary teal
          0x0f4c81, // Secondary navy
          0x36b5cd, // Accent blue
          0x1b263b, // Dark secondary
          0x0d1b2a, // Dark
          0x3a8899, // Medium teal
          0x2a6f97, // Medium blue
          0x1d3557, // Dark blue
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
          
          // Update the loading progress in the HTML
          if (typeof updateLoadingProgress === 'function') {
            updateLoadingProgress(percent);
          } else {
            // If the function doesn't exist, try to manipulate elements directly
            const loadingBar = document.getElementById('loadingBar');
            const loadingPercent = document.getElementById('loadingPercent');
            
            if (loadingBar) loadingBar.style.width = percent + '%';
            if (loadingPercent) loadingPercent.textContent = Math.round(percent) + '%';
          }
        }
      },
      function (error) {
        clearTimeout(loadingTimeout);
        console.error("Error loading brain model:", error);
        createPlaceholderBrain();
        
        // Update the loading screen on error
        if (typeof updateLoadingProgress === 'function') {
          updateLoadingProgress(100);
        } else {
          // If the function doesn't exist, try to manipulate elements directly
          const loadingScreen = document.getElementById('loadingScreen');
          const loadingBar = document.getElementById('loadingBar');
          const loadingPercent = document.getElementById('loadingPercent');
          
          if (loadingBar) loadingBar.style.width = '100%';
          if (loadingPercent) loadingPercent.textContent = '100%';
          
          if (loadingScreen) {
            setTimeout(() => {
              loadingScreen.style.opacity = '0';
              setTimeout(() => {
                loadingScreen.style.display = 'none';
              }, 500);
            }, 300);
          }
        }
      }
    );
  } catch (error) {
    clearTimeout(loadingTimeout);
    console.error("Exception when loading brain model:", error);
    createPlaceholderBrain();
    
    // Update the loading screen on exception
    if (typeof updateLoadingProgress === 'function') {
      updateLoadingProgress(100);
    } else {
      // If the function doesn't exist, try to manipulate elements directly
      const loadingScreen = document.getElementById('loadingScreen');
      const loadingBar = document.getElementById('loadingBar');
      const loadingPercent = document.getElementById('loadingPercent');
      
      if (loadingBar) loadingBar.style.width = '100%';
      if (loadingPercent) loadingPercent.textContent = '100%';
      
      if (loadingScreen) {
        setTimeout(() => {
          loadingScreen.style.opacity = '0';
          setTimeout(() => {
            loadingScreen.style.display = 'none';
          }, 500);
        }, 300);
      }
    }
  }
}

function createTumor() {
  // Remove existing tumor objects if present
  const existingTumor = scene.getObjectByName("tumor");
  if (existingTumor) scene.remove(existingTumor);

  const existingTumorBox = scene.getObjectByName("tumor_box");
  if (existingTumorBox) scene.remove(existingTumorBox);
  
  // Remove existing impact visualization if present
  const existingImpactZone = scene.getObjectByName("impact_zone");
  if (existingImpactZone) scene.remove(existingImpactZone);

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
  
  // Calculate dimensions of the tumor
  const boxWidth = (maxX - minX);
  const boxHeight = (maxY - minY);
  const boxDepth = (maxZ - minZ);
  
  // Calculate tumor volume to determine appropriate rendering method
  const tumorVolume = boxWidth * boxHeight * boxDepth;
  
  // Scale tumor center relative to model scale
  const scaledTumorCenter = tumorBoxCenter.clone().multiplyScalar(modelScale);

  // Create a tumor group to hold all tumor-related objects
  const tumorGroup = new THREE.Group();
  tumorGroup.name = "tumor";
  scene.add(tumorGroup);

  // Create a more realistic, irregular tumor
  // For larger tumors, use multiple overlapping spheres for an irregular shape
  // For small tumors, use a single sphere

  // Main tumor body - create a base ellipsoid matching the dimensions
  const ellipsoidGeometry = new THREE.SphereGeometry(1, 32, 24);
  ellipsoidGeometry.scale(
    boxWidth * modelScale * 1.5,
    boxHeight * modelScale * 1.5, 
    boxDepth * modelScale * 1.5
  );
  
  // Get tumor type from URL for custom rendering
  const urlParams = new URLSearchParams(window.location.search);
  const tumorType = urlParams.get('tumor_type') || 'Unknown';
  
  // Create a material that looks like actual tumor tissue
  // Customize based on tumor type
  let tumorColor = 0xff3333; // Default red
  let tumorEmissive = 0x330000;
  let tumorOpacity = 0.95;
  
  if (tumorType.toLowerCase().includes('glioma')) {
    // Gliomas tend to be more diffuse with irregular borders
    tumorColor = 0xe63946;
    tumorEmissive = 0x330000;
    tumorOpacity = 0.9;
  } else if (tumorType.toLowerCase().includes('meningioma')) {
    // Meningiomas tend to be well-circumscribed
    tumorColor = 0xf4a261;
    tumorEmissive = 0x331100;
    tumorOpacity = 0.95;
  } else if (tumorType.toLowerCase().includes('pituitary')) {
    // Pituitary tumors - make them more visible
    tumorColor = 0x457b9d;
    tumorEmissive = 0x003366; // Brighter blue emission
    tumorOpacity = 1.0; // Fully opaque
    
    // Force position for pituitary tumor to ensure it's in the right place
    scaledTumorCenter.set(0, -0.6 * modelScale, 0.6 * modelScale);
  }
  
  const tumorMaterial = new THREE.MeshStandardMaterial({
    color: tumorColor,
    emissive: tumorEmissive,
    emissiveIntensity: 0.3,
    roughness: 0.7,
    metalness: 0.2,
    transparent: true,
    opacity: tumorOpacity,
  });
  
  const mainTumor = new THREE.Mesh(ellipsoidGeometry, tumorMaterial);
  mainTumor.position.copy(scaledTumorCenter);
  tumorGroup.add(mainTumor);
  
  // Create irregular protrusions on the tumor surface for a more realistic appearance
  // Only for larger tumors
  if (tumorVolume > 0.01) {
    const numProtrusions = Math.floor(tumorVolume * 1000) + 3; // More protrusions for larger tumors
    
    // Create random protrusions
    for (let i = 0; i < numProtrusions; i++) {
      // Random position on the surface of the ellipsoid
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.random() * Math.PI;
      const radius = 1;
      
      // Convert spherical to cartesian coordinates
      const x = radius * Math.sin(phi) * Math.cos(theta);
      const y = radius * Math.sin(phi) * Math.sin(theta);
      const z = radius * Math.cos(phi);
      
      // Scale by the ellipsoid dimensions
      const posX = x * boxWidth * modelScale * 1.5;
      const posY = y * boxHeight * modelScale * 1.5;
      const posZ = z * boxDepth * modelScale * 1.5;
      
      // Size of protrusion varies
      const size = (0.05 + Math.random() * 0.1) * modelScale * Math.min(boxWidth, boxHeight, boxDepth) * 2;
      
      const protrusionGeometry = new THREE.SphereGeometry(size, 8, 8);
      const protrusion = new THREE.Mesh(
        protrusionGeometry,
        tumorMaterial.clone()
      );
      
      // Adjust color slightly for variety
      const hue = Math.random() * 0.1;
      protrusion.material.color.setHSL(0 + hue, 0.8, 0.4);
      
      // Position relative to tumor center
      protrusion.position.set(
        scaledTumorCenter.x + posX,
        scaledTumorCenter.y + posY,
        scaledTumorCenter.z + posZ
      );
      
      tumorGroup.add(protrusion);
    }
  }
  
  // Add subtle pulsing animation to the tumor
  gsap.to(mainTumor.scale, {
    x: 1.05,
    y: 1.05,
    z: 1.05,
    duration: 0.8,
    yoyo: true,
    repeat: -1,
    ease: "sine.inOut"
  });

  // Create a bounding box around the tumor area
  const boxGeometry = new THREE.BoxGeometry(
    boxWidth * modelScale * 3,
    boxHeight * modelScale * 3,
    boxDepth * modelScale * 3
  );
  
  const boxMaterial = new THREE.MeshBasicMaterial({
    color: 0xff5555,
    wireframe: true,
    transparent: true,
    opacity: 0.5,
  });

  const tumorBox = new THREE.Mesh(boxGeometry, boxMaterial);
  tumorBox.name = "tumor_box";
  tumorBox.position.copy(scaledTumorCenter);
  scene.add(tumorBox);
  
  // Create impact visualization
  createImpactZone(scaledTumorCenter, boxWidth, boxHeight, boxDepth);
  
  // Store the tumor type and set visibility of tumor box based on tumor type
  tumorBox.userData.tumorType = tumorType;
  
  // For gliomas, make the tumor box larger to represent infiltration
  if (tumorType.toLowerCase().includes('glioma')) {
    tumorBox.scale.set(1.5, 1.5, 1.5);
  } else if (tumorType.toLowerCase().includes('meningioma')) {
    // For meningiomas, shrink the box to represent well-defined boundary
    tumorBox.scale.set(0.9, 0.9, 0.9);
  }
  
  // Display the tumor coordinates in the console for debugging
  console.log("Tumor created at coordinates:", tumorCoordinates);
  console.log("Tumor center:", tumorBoxCenter);
  console.log("Tumor dimensions:", { width: boxWidth, height: boxHeight, depth: boxDepth });
  console.log("Tumor type:", tumorType);
}

// New function to create the impact zone visualization
function createImpactZone(tumorCenter, width, height, depth) {
  // Remove existing impact zone if present
  const existingImpactZone = scene.getObjectByName("impact_zone");
  if (existingImpactZone) scene.remove(existingImpactZone);
  
  // Create impact zone group
  const impactGroup = new THREE.Group();
  impactGroup.name = "impact_zone";
  
  // Get tumor type from URL for impact analysis
  const urlParams = new URLSearchParams(window.location.search);
  const tumorType = urlParams.get('tumor_type') || 'unknown';
  const brainRegion = urlParams.get('brain_region') || 'unknown';
  
  // Determine impact zone characteristics based on tumor type
  let impactColor = 0xff7777; // Default reddish
  let impactIntensity = 0.3;   // Default medium intensity
  let impactRadius = 2;        // Default multiplier for impact radius
  let impactDistortion = 0.3;  // How uneven the impact zone should be
  let useTractsVisualization = false; // Whether to use tract-based visualization

  // Make the impact characteristics match medical reality based on tumor type
  if (tumorType.toLowerCase().includes('glioma')) {
    // Gliomas tend to be infiltrative with diffuse borders
    impactColor = 0xff5555;
    impactIntensity = 0.5;
    impactRadius = 3.0;
    impactDistortion = 0.5; // More irregular
    useTractsVisualization = true; // Show infiltrative tracts for gliomas
  } else if (tumorType.toLowerCase().includes('meningioma')) {
    // Meningiomas tend to be well-circumscribed and compress rather than infiltrate
    impactColor = 0xffaa77;
    impactIntensity = 0.4;
    impactRadius = 1.8;
    impactDistortion = 0.2; // More regular
    useTractsVisualization = false; // No infiltration for meningiomas
  } else if (tumorType.toLowerCase().includes('pituitary')) {
    // Pituitary tumors tend to be localized but can compress adjacent structures
    impactColor = 0xffcc77;
    impactIntensity = 0.3;
    impactRadius = 1.5;
    impactDistortion = 0.15; // Quite regular
    useTractsVisualization = false; // No infiltration for pituitary tumors
  }
  
  // Create a base impact sphere
  const impactGeometry = new THREE.SphereGeometry(1, 32, 24);
  
  // Calculate the impact zone size based on tumor dimensions and impact radius
  const averageTumorSize = (width + height + depth) / 3;
  impactGeometry.scale(
    (width * modelScale * impactRadius) * (1 + Math.random() * impactDistortion),
    (height * modelScale * impactRadius) * (1 + Math.random() * impactDistortion),
    (depth * modelScale * impactRadius) * (1 + Math.random() * impactDistortion)
  );
  
  // Create a gradient material for the impact zone
  const impactMaterial = new THREE.MeshStandardMaterial({
    color: impactColor,
    emissive: impactColor,
    emissiveIntensity: impactIntensity,
    transparent: true,
    opacity: 0.2,
    side: THREE.DoubleSide,
    depthWrite: false, // Prevents z-fighting and weird rendering artifacts
  });
  
  const impactZone = new THREE.Mesh(impactGeometry, impactMaterial);
  impactZone.position.copy(tumorCenter);
  impactGroup.add(impactZone);
  
  // Add directional impact areas based on tumor location
  // This creates a more anatomically realistic impact that follows white matter tracts
  
  // Get normalized tumor position (0-1 in each dimension)
  const normalizedPos = {
    x: tumorCenter.x / (modelScale * 0.5),
    y: tumorCenter.y / (modelScale * 0.5),
    z: tumorCenter.z / (modelScale * 0.5)
  };
  
  const impactPaths = [];
  
  // Determine which directional impacts to add based on tumor location and brain region
  if (brainRegion.includes("Frontal Lobe")) {
    // Frontal lobe - impact follows fronto-parietal connections
    impactPaths.push({
      direction: new THREE.Vector3(0, -1, 0.3),
      length: 2.5,
      width: 0.6
    });
    // Add connection to corpus callosum for gliomas
    if (useTractsVisualization) {
      impactPaths.push({
        direction: new THREE.Vector3(0, -0.3, -0.8),
        length: 1.8,
        width: 0.5
      });
    }
  } else if (brainRegion.includes("Occipital Lobe")) {
    // Occipital lobe - impact follows visual pathways
    impactPaths.push({
      direction: new THREE.Vector3(0, 0.3, 1), 
      length: 2.2, 
      width: 0.5
    });
    // Add pathway to thalamus for gliomas
    if (useTractsVisualization) {
      impactPaths.push({
        direction: new THREE.Vector3(0.1, 0.6, 0.5),
        length: 2.0,
        width: 0.4
      });
    }
  } else if (brainRegion.includes("Temporal Lobe")) {
    // Temporal lobe - impact follows language or memory pathways
    const isLeft = brainRegion.includes("Left");
    impactPaths.push({
      direction: new THREE.Vector3(isLeft ? 1 : -1, 0.3, 0.2),
      length: 1.8,
      width: 0.7
    });
    // Add connection to hippocampus for gliomas
    if (useTractsVisualization) {
      impactPaths.push({
        direction: new THREE.Vector3(isLeft ? 0.5 : -0.5, -0.5, 0.3),
        length: 1.2,
        width: 0.4
      });
    }
  } else if (brainRegion.includes("Parietal Lobe")) {
    // Parietal lobe - impact follows sensory pathways
    impactPaths.push({ 
      direction: new THREE.Vector3(0, -0.8, 0.6), 
      length: 2.0, 
      width: 0.6 
    });
    // Add connection to motor cortex for gliomas
    if (useTractsVisualization) {
      impactPaths.push({
        direction: new THREE.Vector3(0.3, 0.8, 0.3),
        length: 1.8,
        width: 0.5
      });
    }
  } else if (brainRegion.includes("Cerebellum")) {
    // Cerebellum - impact follows cerebellar peduncles
    impactPaths.push({ 
      direction: new THREE.Vector3(0, 0.8, 0.6), 
      length: 1.5, 
      width: 0.4 
    });
    // Add connection to brain stem
    impactPaths.push({
      direction: new THREE.Vector3(0, 0.2, 0.9),
      length: 1.2,
      width: 0.4
    });
  } else if (brainRegion.includes("Pituitary")) {
    // Pituitary - impact primarily affects optic chiasm and hypothalamus
    impactPaths.push({ 
      direction: new THREE.Vector3(0, 0.5, -0.5), 
      length: 1.0, 
      width: 0.4 
    });
    // Add upward connection to hypothalamus
    impactPaths.push({
      direction: new THREE.Vector3(0, 0.9, 0.1),
      length: 0.8,
      width: 0.3
    });
  } else if (brainRegion.includes("Corpus Callosum")) {
    // Corpus callosum - impact affects interhemispheric connections
    impactPaths.push({ 
      direction: new THREE.Vector3(-0.8, 0, 0), 
      length: 1.8, 
      width: 0.5 
    });
    impactPaths.push({ 
      direction: new THREE.Vector3(0.8, 0, 0), 
      length: 1.8, 
      width: 0.5 
    });
  } else if (brainRegion.includes("Brain Stem")) {
    // Brain stem - various pathways
    impactPaths.push({
      direction: new THREE.Vector3(0, 1, 0),
      length: 1.5,
      width: 0.4
    });
    impactPaths.push({
      direction: new THREE.Vector3(0, -0.5, -0.7),
      length: 1.0,
      width: 0.3
    });
  } else {
    // Default/deep structures - impact radiates in multiple directions
    impactPaths.push({ 
      direction: new THREE.Vector3(0.7, 0.7, 0), 
      length: 1.5, 
      width: 0.5 
    });
    impactPaths.push({ 
      direction: new THREE.Vector3(-0.7, 0.7, 0), 
      length: 1.5, 
      width: 0.5 
    });
    
    // For gliomas, add more paths to show infiltration
    if (useTractsVisualization) {
      impactPaths.push({
        direction: new THREE.Vector3(0.5, -0.7, 0.3),
        length: 1.7,
        width: 0.4
      });
      impactPaths.push({
        direction: new THREE.Vector3(-0.5, -0.7, 0.3),
        length: 1.7,
        width: 0.4
      });
    }
  }
  
  // For gliomas, add special visualization of infiltration
  if (useTractsVisualization) {
    // Add small particles along the impact paths to visualize infiltrating cells
    const particleGeometry = new THREE.SphereGeometry(0.01, 8, 8);
    const particleMaterial = new THREE.MeshStandardMaterial({
      color: impactColor,
      emissive: impactColor,
      emissiveIntensity: 0.7,
      transparent: true,
      opacity: 0.7
    });
    
    // Create particle system for each impact path
    impactPaths.forEach(path => {
      const direction = path.direction.clone().normalize();
      const pathLength = averageTumorSize * modelScale * path.length;
      
      // Create 15-25 particles per path
      const particleCount = 15 + Math.floor(Math.random() * 10);
      
      for (let i = 0; i < particleCount; i++) {
        const particle = new THREE.Mesh(particleGeometry, particleMaterial.clone());
        
        // Position along the path with some randomization
        const distanceFromCenter = (Math.random() * 0.7 + 0.3) * pathLength;
        const randomOffset = new THREE.Vector3(
          (Math.random() - 0.5) * 0.04,
          (Math.random() - 0.5) * 0.04,
          (Math.random() - 0.5) * 0.04
        );
        
        const particlePosition = tumorCenter.clone().add(
          direction.clone().multiplyScalar(distanceFromCenter).add(randomOffset)
        );
        
        particle.position.copy(particlePosition);
        
        // Random size for particles
        const scale = 0.5 + Math.random() * 1.0;
        particle.scale.set(scale, scale, scale);
        
        impactGroup.add(particle);
        
        // Add subtle animation to particles
        gsap.to(particle.position, {
          x: particle.position.x + (Math.random() - 0.5) * 0.02,
          y: particle.position.y + (Math.random() - 0.5) * 0.02,
          z: particle.position.z + (Math.random() - 0.5) * 0.02,
          duration: 2 + Math.random() * 2,
          yoyo: true,
          repeat: -1,
          ease: "sine.inOut"
        });
      }
    });
  }
  
  // Create the directional impact paths
  impactPaths.forEach((path, index) => {
    // Normalize the direction vector
    const dir = path.direction.normalize();
    
    // Create a tapered cylinder for the impact path
    const pathLength = averageTumorSize * modelScale * path.length;
    const baseRadius = averageTumorSize * modelScale * path.width;
    const topRadius = baseRadius * 0.4; // Tapered end
    
    const pathGeometry = new THREE.CylinderGeometry(
      topRadius, baseRadius, pathLength, 16, 1, true
    );
    
    // Rotate cylinder to align with direction
    pathGeometry.rotateX(Math.PI / 2);
    
    // Create a material with gradient opacity
    const pathMaterial = new THREE.MeshStandardMaterial({
      color: impactColor,
      emissive: impactColor,
      emissiveIntensity: impactIntensity * 0.7,
      transparent: true,
      opacity: 0.15,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    
    const pathMesh = new THREE.Mesh(pathGeometry, pathMaterial);
    
    // Position the impact path - offset from tumor center in the direction
    const offset = dir.clone().multiplyScalar(pathLength * 0.5);
    pathMesh.position.copy(tumorCenter.clone().add(offset));
    
    // Orient the impact path to point in the direction
    pathMesh.lookAt(tumorCenter.clone().add(dir.clone().multiplyScalar(pathLength)));
    
    // Add to impact group
    impactGroup.add(pathMesh);
    
    // Add subtle animation to the impact paths
    gsap.to(pathMesh.material, {
      opacity: 0.08,
      duration: 2 + index * 0.5,
      yoyo: true,
      repeat: -1,
      ease: "sine.inOut"
    });
  });
  
  // Add subtle pulse animation to the impact zone
  gsap.to(impactZone.scale, {
    x: 1.05,
    y: 1.05,
    z: 1.05,
    duration: 3,
    yoyo: true,
    repeat: -1,
    ease: "sine.inOut"
  });
  
  // Create a label for the impact zone
  if (tumorType !== 'unknown') {
    const impactTypeEl = document.createElement('div');
    impactTypeEl.className = 'impact-type-label';
    impactTypeEl.style.position = 'absolute';
    impactTypeEl.style.top = '70px';
    impactTypeEl.style.left = '50%';
    impactTypeEl.style.transform = 'translateX(-50%)';
    impactTypeEl.style.background = 'rgba(0,0,0,0.7)';
    impactTypeEl.style.color = 'white';
    impactTypeEl.style.padding = '5px 10px';
    impactTypeEl.style.borderRadius = '5px';
    impactTypeEl.style.fontSize = '14px';
    impactTypeEl.style.zIndex = '1002';
    
    // Set the impact type text based on tumor type
    if (tumorType.toLowerCase().includes('glioma')) {
      impactTypeEl.textContent = 'Infiltrative Pattern (Glioma)';
      impactTypeEl.style.background = 'rgba(220, 53, 69, 0.8)';
    } else if (tumorType.toLowerCase().includes('meningioma')) {
      impactTypeEl.textContent = 'Compressive Effect (Meningioma)';
      impactTypeEl.style.background = 'rgba(255, 193, 7, 0.8)';
    } else if (tumorType.toLowerCase().includes('pituitary')) {
      impactTypeEl.textContent = 'Localized Impact (Pituitary)';
      impactTypeEl.style.background = 'rgba(23, 162, 184, 0.8)';
    } else {
      impactTypeEl.textContent = 'Tumor Impact Zone';
    }
    
    // Remove any existing impact type labels
    const existingLabel = document.querySelector('.impact-type-label');
    if (existingLabel) {
      existingLabel.remove();
    }
    
    document.body.appendChild(impactTypeEl);
  }
  
  scene.add(impactGroup);
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

// Update the toggleTumorAxis function to show coordinates in the bottom left
function toggleTumorAxis() {
  const axis = scene.getObjectByName("tumor_axis");
  const xLabel = scene.getObjectByName("x_label");
  const yLabel = scene.getObjectByName("y_label");
  const zLabel = scene.getObjectByName("z_label");
  const coordinatesDisplay = document.getElementById("coordinates-display");

  if (axis) {
    axis.visible = !axis.visible;
    xLabel.visible = axis.visible;
    yLabel.visible = axis.visible;
    zLabel.visible = axis.visible;

    if (axis.visible) {
      // Display tumor coordinates in the dedicated container
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

      coordinatesDisplay.innerHTML = html;
      coordinatesDisplay.style.display = "block";
    } else {
      // Hide the coordinates display when axis is hidden
      coordinatesDisplay.style.display = "none";
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
        // Toggle affected regions if we're holding the Shift key
        if (event.shiftKey) {
          const toggleButton = document.getElementById('toggleAffectedRegions');
          if (toggleButton) toggleButton.click();
        } else {
          resetView();
        }
        break;
      case " ":
        event.preventDefault();
        toggleExplode();
        break;
      case "a":
      case "A":
        toggleTumorAxis();
        break;
      case "h":
      case "H":
        // Shortcut to toggle highlighting affected regions
        const toggleButton = document.getElementById('toggleAffectedRegions');
        if (toggleButton) toggleButton.click();
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

// Function to handle toggling affected regions
function toggleAffectedRegions() {
  const model = scene.getObjectByName("brain_model");
  if (!model) return;
  
  // Get all brain parts
  const brainParts = [];
  model.traverse((child) => {
    if (child.isMesh) {
      brainParts.push(child);
    }
  });
  
  // Check if we're in affected regions mode
  const affectedRegionsMode = model.userData.affectedRegionsMode || false;
  
  if (!affectedRegionsMode) {
    // Enable affected regions mode - set all parts to same color
    model.userData.originalMaterials = {};
    brainParts.forEach(part => {
      // Store original material for restoration
      model.userData.originalMaterials[part.uuid] = part.material.clone();
      
      // Create new highlighted material
      const highlightMaterial = new THREE.MeshStandardMaterial({
        color: 0xf08080, // Light coral color for affected regions
        emissive: 0xff4d4d,
        emissiveIntensity: 0.3,
        roughness: 0.7,
        metalness: 0.2
      });
      
      // Apply the new material
      part.material = highlightMaterial;
    });
    
    // Set flag to true
    model.userData.affectedRegionsMode = true;
    
    // Update button text if it exists
    const toggleButton = document.getElementById('toggleAffectedRegions');
    if (toggleButton) {
      toggleButton.textContent = 'Hide Affected Regions';
      toggleButton.classList.add('active');
    }
  } else {
    // Disable affected regions mode - restore original materials
    brainParts.forEach(part => {
      if (model.userData.originalMaterials && model.userData.originalMaterials[part.uuid]) {
        part.material = model.userData.originalMaterials[part.uuid];
      }
    });
    
    // Set flag to false
    model.userData.affectedRegionsMode = false;
    
    // Update button text if it exists
    const toggleButton = document.getElementById('toggleAffectedRegions');
    if (toggleButton) {
      toggleButton.textContent = 'Affected Regions';
      toggleButton.classList.remove('active');
    }
  }
}

// Function to toggle between functional and standard view
function toggleFunctionalView() {
  const functionalViewActive = document.body.classList.contains('functional-view') || 
                               document.body.classList.contains('functional-view-active');
  
  if (functionalViewActive) {
    document.body.classList.remove('functional-view');
    document.body.classList.remove('functional-view-active');
    document.getElementById('toggleFunctionalView').textContent = 'Functional View';
    document.getElementById('toggleFunctionalView').classList.remove('active');
  } else {
    document.body.classList.add('functional-view');
    document.body.classList.add('functional-view-active');
    document.getElementById('toggleFunctionalView').textContent = 'Standard View';
    document.getElementById('toggleFunctionalView').classList.add('active');
    
    // Position highlights if they exist
    const motorHighlight = document.getElementById('motorHighlight');
    const visualHighlight = document.getElementById('visualHighlight');
    const languageHighlight = document.getElementById('languageHighlight');
    const vesselHighlight = document.getElementById('vesselHighlight');
    
    if (motorHighlight && visualHighlight && languageHighlight && vesselHighlight) {
      // Position highlights based on brain model positioning
      motorHighlight.style.left = '40%';
      motorHighlight.style.top = '35%';
      
      visualHighlight.style.left = '70%';
      visualHighlight.style.top = '45%';
      
      languageHighlight.style.left = '30%';
      languageHighlight.style.top = '45%';
      
      vesselHighlight.style.left = '50%';
      vesselHighlight.style.top = '55%';
    } else {
      // Create highlights if they don't exist
      createFunctionalHighlights();
    }
  }
  
  // Disable affected regions mode if active
  const model = scene.getObjectByName("brain_model");
  if (model && model.userData.affectedRegionsMode) {
    toggleAffectedRegions();
  }
}

// Function to create functional highlights if they don't exist
function createFunctionalHighlights() {
  // Create overlay
  let overlay = document.getElementById('functionalViewOverlay');
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.id = 'functionalViewOverlay';
    overlay.className = 'functional-view-overlay';
    document.body.appendChild(overlay);
  }
  
  // Create highlights for different functional areas
  createHighlight('motorHighlight', 'motor-highlight', '40%', '35%');
  createHighlight('visualHighlight', 'visual-highlight', '70%', '45%');
  createHighlight('languageHighlight', 'language-highlight', '30%', '45%');
  createHighlight('vesselHighlight', 'vessel-highlight', '50%', '55%');
}

// Helper function to create a highlight element
function createHighlight(id, className, left, top) {
  let highlight = document.getElementById(id);
  if (!highlight) {
    highlight = document.createElement('div');
    highlight.id = id;
    highlight.className = 'structure-highlight ' + className;
    highlight.style.left = left;
    highlight.style.top = top;
    document.body.appendChild(highlight);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  // Add enhanced styling for better visual appeal
  const enhancedStyles = document.createElement('style');
  enhancedStyles.textContent = `
    body {
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
      color: #e3e3e3;
      font-family: 'Roboto', 'Segoe UI', Arial, sans-serif;
      animation: fadeIn 1s ease-out;
    }
    
    /* Modern button styling */
    button {
      background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%) !important;
      border: none !important;
      color: white !important;
      padding: 10px 20px !important;
      border-radius: 8px !important;
      font-weight: 600 !important;
      letter-spacing: 0.5px !important;
      transition: all 0.3s ease !important;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2) !important;
      position: relative;
      overflow: hidden;
    }
    
    button:hover {
      transform: translateY(-3px) !important;
      box-shadow: 0 7px 14px rgba(0, 0, 0, 0.3) !important;
    }
    
    button:active {
      transform: translateY(1px) !important;
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Button specific colors */
    .explode-btn {
      background: linear-gradient(135deg, #d90429 0%, #ef233c 100%) !important;
    }
    
    .explode-btn.reformed {
      background: linear-gradient(135deg, #2a9d8f 0%, #52b788 100%) !important;
    }
    
    .axis-btn {
      background: linear-gradient(135deg, #3a86ff 0%, #4361ee 100%) !important;
    }
    
    .materials-btn {
      background: linear-gradient(135deg, #7209b7 0%, #9d4edd 100%) !important;
    }
    
    /* Enhanced panel styling */
    .anatomical-panel {
      background: rgba(27, 38, 59, 0.9) !important;
      box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3) !important;
      border-radius: 12px !important;
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .anatomical-panel h3, .anatomical-panel .panel-title {
      color: #90e0ef !important;
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      padding-bottom: 8px;
      margin-bottom: 15px;
    }
    
    .anatomical-panel .panel-subtitle {
      color: #caf0f8 !important;
    }
    
    /* Toggle buttons within panel */
    .toggle-button {
      background: linear-gradient(135deg, #5e60ce 0%, #4ea8de 100%) !important;
      border-radius: 8px !important;
      border: none !important;
      transition: all 0.3s ease !important;
      position: relative;
      overflow: hidden;
    }
    
    .toggle-button:before {
      content: '';
      position: absolute;
      top: -50%;
      left: -50%;
      width: 200%;
      height: 200%;
      background: rgba(255, 255, 255, 0.1);
      transform: rotate(30deg);
      transition: all 0.5s ease;
      opacity: 0;
    }
    
    .toggle-button:hover:before {
      opacity: 1;
    }
    
    .toggle-button.active {
      background: linear-gradient(135deg, #4cc9f0 0%, #4361ee 100%) !important;
      box-shadow: 0 0 15px rgba(76, 201, 240, 0.5) !important;
    }
    
    /* Enhanced structure indicators */
    .structure-indicator {
      box-shadow: inset 0 0 5px rgba(0, 0, 0, 0.3) !important;
      transition: all 0.3s ease !important;
    }
    
    .distance-safe {
      background: linear-gradient(135deg, #52b788 0%, #2a9d8f 100%) !important;
    }
    
    .distance-close {
      background: linear-gradient(135deg, #ff9e00 0%, #ff7b00 100%) !important;
    }
    
    .distance-critical {
      background: linear-gradient(135deg, #d90429 0%, #ef233c 100%) !important;
    }
    
    /* Enhanced highlights */
    .structure-highlight {
      box-shadow: 0 0 20px 10px rgba(255, 255, 255, 0.3) !important;
      animation: pulse-highlight 2s infinite alternate;
    }
    
    @keyframes pulse-highlight {
      0% { box-shadow: 0 0 20px 5px rgba(255, 255, 255, 0.3); }
      100% { box-shadow: 0 0 30px 10px rgba(255, 255, 255, 0.5); }
    }
    
    /* Coordinate form styling */
    .coordinate-form {
      background: rgba(27, 38, 59, 0.95) !important;
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px !important;
      box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3) !important;
    }
    
    .coordinate-form input {
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.2);
      color: #e3e3e3;
      border-radius: 6px;
      padding: 8px 12px;
      transition: all 0.3s ease;
    }
    
    .coordinate-form input:focus {
      background: rgba(255, 255, 255, 0.15);
      border-color: rgba(255, 255, 255, 0.3);
      outline: none;
      box-shadow: 0 0 0 3px rgba(76, 201, 240, 0.3);
    }
    
    /* Enhanced notifications */
    #notification-banner {
      background: rgba(27, 38, 59, 0.95) !important;
      backdrop-filter: blur(8px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
      border-radius: 8px;
      animation: slideDown 0.5s ease forwards;
    }
    
    @keyframes slideDown {
      from { transform: translateY(-20px); opacity: 0; }
      to { transform: translateY(0); opacity: 1; }
    }
    
    /* Enhanced tooltips */
    #label, .region-tooltip {
      background: rgba(27, 38, 59, 0.95) !important;
      backdrop-filter: blur(8px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
      border-radius: 8px;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Enhanced labels */
    #distanceIndicator, #tumor-coords, #tumorTypeLabel {
      background: rgba(27, 38, 59, 0.95) !important;
      backdrop-filter: blur(8px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
      border-radius: 8px;
      font-family: 'Roboto Mono', monospace;
      letter-spacing: 0.5px;
    }
    
    /* Loading animation */
    .loading {
      background: rgba(27, 38, 59, 0.95) !important;
      backdrop-filter: blur(10px);
      animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
      0% { box-shadow: 0 0 0 0 rgba(76, 201, 240, 0.4); }
      70% { box-shadow: 0 0 0 20px rgba(76, 201, 240, 0); }
      100% { box-shadow: 0 0 0 0 rgba(76, 201, 240, 0); }
    }
    
    /* Improved progress bar */
    #loadingProgress {
      background: rgba(27, 38, 59, 0.5);
      border-radius: 10px;
      overflow: hidden;
    }
    
    #progressBar {
      background: linear-gradient(90deg, #4cc9f0, #4361ee);
      transition: width 0.3s ease;
      border-radius: 10px;
      box-shadow: 0 0 10px rgba(76, 201, 240, 0.5);
    }
    
    /* Enhanced affected regions */
    .affected-regions-active .blink-border {
      border-color: rgba(255, 76, 76, 0.5) !important;
      box-shadow: inset 0 0 30px rgba(255, 76, 76, 0.2);
    }
    
    /* Page transitions */
    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }
    
    .ripple {
      position: absolute;
      border-radius: 50%;
      transform: scale(0);
      animation: ripple 0.6s linear;
      background-color: rgba(255, 255, 255, 0.3);
      pointer-events: none;
    }
    
    @keyframes ripple {
      to {
        transform: scale(4);
        opacity: 0;
      }
    }
    
    /* Custom cursor */
    .cursor-effect {
      pointer-events: none;
      position: fixed;
      border-radius: 50%;
      transform: translate(-50%, -50%);
      mix-blend-mode: difference;
      z-index: 9999;
      transition: opacity 0.3s ease, width 0.2s ease, height 0.2s ease;
    }
    
    /* Loading spinner animation */
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
    
    .explode-burst {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      border-radius: 50%;
      background-color: rgba(255, 255, 255, 0.8);
      z-index: -1;
    }
  `;
  document.head.appendChild(enhancedStyles);

  // Add CSS styles for functional view elements
  const style = document.createElement('style');
  style.textContent = `
    /* Functional View Styles */
    .functional-view-overlay {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      background: radial-gradient(circle, rgba(0,0,0,0) 50%, rgba(100,0,100,0.1) 100%);
      z-index: 800;
      opacity: 0;
      transition: opacity 0.5s ease;
    }
    
    .functional-view-active .functional-view-overlay {
      opacity: 1;
    }
    
    .functional-view-active #tumorTypeLabel {
      background: rgba(111, 66, 193, 0.8);
    }
    
    .structure-highlight {
      position: fixed;
      border-radius: 50%;
      pointer-events: none;
      z-index: 850;
      box-shadow: 0 0 15px 5px rgba(255,255,255,0.7);
      opacity: 0;
      transition: opacity 0.5s ease;
    }
    
    .functional-view-active .structure-highlight {
      opacity: 1;
    }
    
    .motor-highlight {
      background-color: rgba(255, 99, 132, 0.3);
      width: 50px;
      height: 50px;
    }
    
    .visual-highlight {
      background-color: rgba(54, 162, 235, 0.3);
      width: 40px;
      height: 40px;
    }
    
    .language-highlight {
      background-color: rgba(255, 206, 86, 0.3);
      width: 45px;
      height: 45px;
    }
    
    .vessel-highlight {
      background-color: rgba(75, 192, 192, 0.3);
      width: 30px;
      height: 30px;
    }
  `;
  document.head.appendChild(style);

  // Create ripple effect for buttons
  function addRippleEffect(button) {
    button.addEventListener('click', function(e) {
      const ripple = document.createElement('span');
      const rect = button.getBoundingClientRect();
      
      const size = Math.max(rect.width, rect.height);
      const x = e.clientX - rect.left - size / 2;
      const y = e.clientY - rect.top - size / 2;
      
      ripple.style.width = ripple.style.height = `${size}px`;
      ripple.style.left = `${x}px`;
      ripple.style.top = `${y}px`;
      ripple.className = 'ripple';
      
      button.appendChild(ripple);
      
      setTimeout(() => {
        if (ripple.parentNode) {
          ripple.parentNode.removeChild(ripple);
        }
      }, 600);
    });
  }
  
  // Create custom cursor effect
  const cursorEffect = document.createElement('div');
  cursorEffect.className = 'cursor-effect';
  cursorEffect.style.width = '20px';
  cursorEffect.style.height = '20px';
  cursorEffect.style.border = '2px solid rgba(76, 201, 240, 0.6)';
  cursorEffect.style.opacity = '0';
  document.body.appendChild(cursorEffect);
  
  document.addEventListener('mousemove', (e) => {
    if (modelLoaded) {
      cursorEffect.style.opacity = '1';
      cursorEffect.style.left = `${e.clientX}px`;
      cursorEffect.style.top = `${e.clientY}px`;
    }
  });
  
  document.addEventListener('mousedown', () => {
    if (modelLoaded) {
      cursorEffect.style.width = '15px';
      cursorEffect.style.height = '15px';
      cursorEffect.style.borderColor = 'rgba(76, 201, 240, 0.9)';
    }
  });
  
  document.addEventListener('mouseup', () => {
    if (modelLoaded) {
      cursorEffect.style.width = '20px';
      cursorEffect.style.height = '20px';
      cursorEffect.style.borderColor = 'rgba(76, 201, 240, 0.6)';
    }
  });
  
  document.addEventListener('mouseleave', () => {
    cursorEffect.style.opacity = '0';
  });
  
  // Initialize coordinate form elements
  coordinateForm = document.getElementById("tumorCoordinateForm");
  createTumorBtn = document.getElementById("createTumorBtn");
  useDefaultCoordinatesBtn = document.getElementById("useDefaultCoordinatesBtn");
  coordinateError = document.getElementById("coordinateError");
  changeTumorBtn = document.getElementById("changeTumorBtn");
  
  // Set up event listeners for coordinate form
  createTumorBtn.addEventListener("click", handleCoordinateSubmit);
  useDefaultCoordinatesBtn.addEventListener("click", () => {
    fillDefaultCoordinates();
    handleCoordinateSubmit();
  });
  changeTumorBtn.addEventListener("click", showCoordinateForm);
  
  // Enhance the explode button functionality
  const originalToggleExplode = toggleExplode;
  toggleExplode = function() {
    originalToggleExplode();
    const explodeButton = document.querySelector(".explode-btn");
    
    if (explodeButton) {
      if (window.forceExplode) {
        explodeButton.classList.remove('reformed');
        // Add burst animation
        const burst = document.createElement('div');
        burst.className = 'explode-burst';
        burst.style.width = '10px';
        burst.style.height = '10px';
        explodeButton.appendChild(burst);
        
        gsap.to(burst, {
          width: '300px',
          height: '300px',
          opacity: 0,
          duration: 0.8,
          onComplete: () => {
            if (burst.parentNode) {
              burst.parentNode.removeChild(burst);
            }
          }
        });
      } else {
        explodeButton.classList.add('reformed');
      }
    }
  };
  
  // Create top button container
  const topButtonContainer = document.createElement("div");
  topButtonContainer.className = "top-button-container";
  topButtonContainer.style.position = "absolute";
  topButtonContainer.style.top = "20px";
  topButtonContainer.style.left = "50%";
  topButtonContainer.style.transform = "translateX(-50%)";
  topButtonContainer.style.display = "flex";
  topButtonContainer.style.flexWrap = "wrap";
  topButtonContainer.style.justifyContent = "center";
  topButtonContainer.style.gap = "10px";
  topButtonContainer.style.zIndex = "1000";
  topButtonContainer.style.maxWidth = "90%";
  document.body.appendChild(topButtonContainer);
  
  // UI elements for brain controls
  const explodeButton = document.createElement("button");
  explodeButton.textContent = "REFORM BRAIN";
  explodeButton.className = "explode-btn reformed";
  explodeButton.style.padding = "10px 20px";
  explodeButton.style.color = "white";
  explodeButton.style.border = "none";
  explodeButton.style.borderRadius = "5px";
  explodeButton.style.cursor = "pointer";
  explodeButton.onclick = toggleExplode;
  addRippleEffect(explodeButton);
  topButtonContainer.appendChild(explodeButton);

  const axisButton = document.createElement("button");
  axisButton.textContent = "SHOW AXIS";
  axisButton.className = "axis-btn";
  axisButton.style.padding = "10px 20px";
  axisButton.style.color = "white";
  axisButton.style.border = "none";
  axisButton.style.borderRadius = "5px";
  axisButton.style.cursor = "pointer";
  axisButton.onclick = toggleTumorAxis;
  addRippleEffect(axisButton);
  topButtonContainer.appendChild(axisButton);

  // Add materials adjustment button
  const materialsButton = document.createElement("button");
  materialsButton.textContent = "ENHANCE MATERIALS";
  materialsButton.className = "materials-btn";
  materialsButton.style.padding = "10px 20px";
  materialsButton.style.color = "white";
  materialsButton.style.border = "none";
  materialsButton.style.borderRadius = "5px";
  materialsButton.style.cursor = "pointer";
  materialsButton.onclick = adjustModelMaterials;
  addRippleEffect(materialsButton);
  topButtonContainer.appendChild(materialsButton);
  
  // Create a right-side anatomical analysis panel
  const anatomicalPanel = document.createElement("div");
  anatomicalPanel.className = "anatomical-panel";
  anatomicalPanel.style.position = "absolute";
  anatomicalPanel.style.top = "70px";
  anatomicalPanel.style.right = "20px";
  anatomicalPanel.style.width = "300px";
  anatomicalPanel.style.background = "rgba(31, 31, 43, 0.9)";
  anatomicalPanel.style.color = "white";
  anatomicalPanel.style.padding = "15px";
  anatomicalPanel.style.borderRadius = "5px";
  anatomicalPanel.style.zIndex = "1000";
  anatomicalPanel.style.maxHeight = "80vh";
  anatomicalPanel.style.overflow = "auto";
  
  anatomicalPanel.innerHTML = `
    <h3 style="margin-top:0">Anatomical Analysis</h3>
    <div class="region-info">
      <h4>Region: Pituitary Fossa</h4>
      <h4>Proximity to Critical Structures:</h4>
      <div class="structure-item">
        <span class="status-indicator" style="display:inline-block;width:12px;height:12px;background:#28a745;border-radius:50%;margin-right:5px;"></span>
        <strong>Motor Pathway</strong> - Safe distance
      </div>
      <div class="structure-item">
        <span class="status-indicator" style="display:inline-block;width:12px;height:12px;background:#ffc107;border-radius:50%;margin-right:5px;"></span>
        <strong>Visual Pathway</strong> - Close proximity
      </div>
      <div class="structure-item">
        <span class="status-indicator" style="display:inline-block;width:12px;height:12px;background:#28a745;border-radius:50%;margin-right:5px;"></span>
        <strong>Language Area</strong> - Safe distance
      </div>
      <div class="structure-item">
        <span class="status-indicator" style="display:inline-block;width:12px;height:12px;background:#dc3545;border-radius:50%;margin-right:5px;"></span>
        <strong>Major Blood Vessels</strong> - Critical proximity
      </div>
    </div>
    <div class="impact-assessment">
      <h4>Impact Assessment</h4>
      <div class="impact-item">
        <strong style="color:#4cd964">Impact on Hormone Regulation, Visual Pathways</strong>
      </div>
      <h4>Functional Impact:</h4>
      <p>This pituitary in the Pituitary Fossa may affect Hormone Regulation, Visual Pathways. Additional risks include: Compression of optic chiasm, Disruption of hormone production.</p>
    </div>
  `;
  
  document.body.appendChild(anatomicalPanel);
  
  // Create coordinates display container (initially hidden)
  const coordinatesDisplay = document.createElement("div");
  coordinatesDisplay.id = "coordinates-display";
  coordinatesDisplay.style.position = "absolute";
  coordinatesDisplay.style.bottom = "20px";
  coordinatesDisplay.style.left = "20px";
  coordinatesDisplay.style.background = "rgba(0,0,0,0.7)";
  coordinatesDisplay.style.color = "white";
  coordinatesDisplay.style.padding = "10px";
  coordinatesDisplay.style.borderRadius = "5px";
  coordinatesDisplay.style.fontFamily = "monospace";
  coordinatesDisplay.style.zIndex = "1000";
  coordinatesDisplay.style.display = "none";
  document.body.appendChild(coordinatesDisplay);

  const tooltip = document.createElement("div");
  tooltip.style.position = "absolute";
  tooltip.style.top = "50%";
  tooltip.style.left = "50%";
  tooltip.style.transform = "translate(-50%, -50%)";
  tooltip.style.background = "rgba(0,0,0,0.8)";
  tooltip.style.color = "white";
  tooltip.style.padding = "20px";
  tooltip.style.borderRadius = "10px";
  tooltip.style.textAlign = "center";
  tooltip.style.zIndex = "2000";
  tooltip.innerHTML =
    "<h3>3D Brain Model with Tumor Visualization</h3><p>Enter tumor coordinates or use defaults</p><p>Double-click to focus on tumor</p><p>Press SPACE to explode brain</p>";
  document.body.appendChild(tooltip);

  setTimeout(() => {
    gsap.to(tooltip, {
      opacity: 0,
      duration: 1,
      onComplete: () => document.body.removeChild(tooltip),
    });
  }, 4000);

  // Start with coordinate form visible, brain will load after form submission
  coordinateForm.style.display = "block";
  changeTumorBtn.style.display = "none";
  setupEventHandlers();
  animate();
  
  // Make sure all buttons have the same width and styling
  const buttons = [explodeButton, axisButton, materialsButton];
  
  buttons.forEach(button => {
    button.style.minWidth = "180px";
    button.style.textAlign = "center";
    button.style.fontWeight = "bold";
    button.style.margin = "0 5px";
  });
  
  // Add a beautiful loading screen
  if (!modelLoaded) {
    const loadingScreen = document.createElement('div');
    loadingScreen.className = 'enhanced-loading';
    loadingScreen.style.position = 'fixed';
    loadingScreen.style.top = '0';
    loadingScreen.style.left = '0';
    loadingScreen.style.width = '100%';
    loadingScreen.style.height = '100%';
    loadingScreen.style.background = 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)';
    loadingScreen.style.display = 'flex';
    loadingScreen.style.flexDirection = 'column';
    loadingScreen.style.justifyContent = 'center';
    loadingScreen.style.alignItems = 'center';
    loadingScreen.style.zIndex = '9999';
    
    const loadingTitle = document.createElement('h1');
    loadingTitle.textContent = 'Brain Tumor Visualization';
    loadingTitle.style.color = '#caf0f8';
    loadingTitle.style.fontFamily = "'Roboto', sans-serif";
    loadingTitle.style.fontSize = '2.5rem';
    loadingTitle.style.marginBottom = '20px';
    loadingTitle.style.textAlign = 'center';
    loadingTitle.style.opacity = '0';
    loadingScreen.appendChild(loadingTitle);
    
    const loadingSubtitle = document.createElement('div');
    loadingSubtitle.textContent = 'Advanced 3D Neurological Analysis';
    loadingSubtitle.style.color = '#90e0ef';
    loadingSubtitle.style.fontFamily = "'Roboto', sans-serif";
    loadingSubtitle.style.fontSize = '1.2rem';
    loadingSubtitle.style.marginBottom = '50px';
    loadingSubtitle.style.textAlign = 'center';
    loadingSubtitle.style.opacity = '0';
    loadingScreen.appendChild(loadingSubtitle);
    
    const spinnerContainer = document.createElement('div');
    spinnerContainer.style.position = 'relative';
    spinnerContainer.style.width = '100px';
    spinnerContainer.style.height = '100px';
    spinnerContainer.style.opacity = '0';
    
    const spinner = document.createElement('div');
    spinner.style.border = '4px solid rgba(255, 255, 255, 0.1)';
    spinner.style.borderTop = '4px solid #4cc9f0';
    spinner.style.borderRadius = '50%';
    spinner.style.width = '100%';
    spinner.style.height = '100%';
    spinner.style.animation = 'spin 1s linear infinite';
    spinnerContainer.appendChild(spinner);
    
    const spinnerInner = document.createElement('div');
    spinnerInner.style.position = 'absolute';
    spinnerInner.style.top = '20px';
    spinnerInner.style.left = '20px';
    spinnerInner.style.right = '20px';
    spinnerInner.style.bottom = '20px';
    spinnerInner.style.border = '4px solid rgba(255, 255, 255, 0.1)';
    spinnerInner.style.borderTop = '4px solid #90e0ef';
    spinnerInner.style.borderRadius = '50%';
    spinnerInner.style.animation = 'spin 1.5s linear infinite reverse';
    spinnerContainer.appendChild(spinnerInner);
    
    loadingScreen.appendChild(spinnerContainer);
    document.body.appendChild(loadingScreen);
    
    // Animate loading screen elements
    gsap.to(loadingTitle, { opacity: 1, duration: 0.8, delay: 0.2 });
    gsap.to(loadingSubtitle, { opacity: 1, duration: 0.8, delay: 0.5 });
    gsap.to(spinnerContainer, { opacity: 1, duration: 0.8, delay: 0.8 });
    
    // Remove loading screen when model is loaded
    const originalLoadBrainModel = loadBrainModel;
    loadBrainModel = function() {
      originalLoadBrainModel();
      setTimeout(() => {
        gsap.to(loadingScreen, { 
          opacity: 0, 
          duration: 0.8, 
          onComplete: () => {
            if (document.body.contains(loadingScreen)) {
              document.body.removeChild(loadingScreen);
            }
          }
        });
      }, 1500);
    };
  }
  
  // Add 3D depth effect
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  
  // Add ripple effect to all toggle buttons
  document.querySelectorAll('.toggle-button').forEach(button => {
    addRippleEffect(button);
  });
});
