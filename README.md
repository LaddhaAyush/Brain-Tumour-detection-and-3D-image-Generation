# Brain Tumor Mapping Application

## Overview
This application provides an interactive 3D visualization of a brain model with customizable tumor mapping. It allows users to input specific tumor coordinates or use default values.

## Features
- Input custom tumor coordinates (8 points to define a bounding box)
- Default tumor coordinates for demonstration
- Interactive 3D brain model visualization
- Brain explosion effect for better visualization
- Tumor highlighting and coordinate display
- Keyboard controls for navigation and interaction

## How to Use

### Setting Up Tumor Coordinates
1. When you first open the application, you'll see a form to input tumor coordinates
2. Enter X, Y, Z values for all 8 points of the tumor bounding box
3. Click "Create Tumor" to visualize with your coordinates
4. Alternatively, click "Use Default Coordinates" to use pre-defined values

### Interacting with the Model
- **Mouse Controls**:
  - Left-click and drag to rotate the brain model
  - Scroll to zoom in/out
  - Double-click to focus on tumor

- **Keyboard Controls**:
  - **Space**: Toggle brain explosion effect
  - **A**: Toggle tumor axis and coordinate display
  - **+/-**: Increase/decrease brain size
  - **R**: Reset view
  - **Tab**: Cycle through brain parts

- **Buttons**:
  - "EXPLODE BRAIN": Manually toggle the explosion effect
  - "SHOW AXIS": Display the tumor coordinate axes
  - "ENHANCE MATERIALS": Improve visualization materials
  - "Change Tumor Location": Return to the coordinate input form

## Tumor Coordinate Format
Tumor coordinates should define a 3D bounding box with 8 points. The suggested order is:
1. Front bottom left
2. Front bottom right
3. Front top right
4. Front top left
5. Back bottom left
6. Back bottom right
7. Back top right
8. Back top left

## Technical Details
This application uses:
- Three.js for 3D rendering
- GSAP for animations
- GLTFLoader for loading 3D models

## Notes
- The brain model will be automatically loaded after tumor coordinates are submitted
- If the model fails to load, a simple placeholder brain will be displayed
- The "Change Tumor Location" button allows you to update coordinates during the session 