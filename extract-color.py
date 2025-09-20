import cv2
import numpy as np
from collections import defaultdict
import colorsys

# Add these new constants at the top of your code
COLOR_SIMILARITY_THRESHOLD = 0.15  # Lower = more strict color matching
NOSE_DISTANCE_THRESHOLD = 150     # Max pixels nose can move between frames
COLOR_WEIGHT = 0.4               # Weight for color similarity (0-1)
POSITION_WEIGHT = 0.6            # Weight for position similarity (0-1)
MIN_CLOTHING_AREA = 100          # Minimum area for clothing color extraction

# Global storage for person colors
person_colors = {}  # {person_id: {"torso_color": (r,g,b), "confidence": float}}

def extract_dominant_color(image_region, mask=None):
    """Extract the dominant color from an image region"""
    if image_region is None or image_region.size == 0:
        return None
    
    # Apply mask if provided
    if mask is not None:
        image_region = cv2.bitwise_and(image_region, image_region, mask=mask)
    
    # Reshape to list of pixels
    pixels = image_region.reshape(-1, 3)
    
    # Remove very dark pixels (likely background/shadows)
    pixels = pixels[np.sum(pixels, axis=1) > 30]
    
    if len(pixels) == 0:
        return None
    
    # Use k-means clustering to find dominant colors
    from sklearn.cluster import KMeans
    try:
        # Try 3 clusters first
        kmeans = KMeans(n_clusters=min(3, len(pixels)), random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get the most frequent cluster (dominant color)
        labels = kmeans.labels_
        counts = np.bincount(labels)
        dominant_color = kmeans.cluster_centers_[np.argmax(counts)]
        
        return tuple(map(int, dominant_color))
    except:
        # Fallback to mean color
        return tuple(map(int, np.mean(pixels, axis=0)))

def get_torso_region(frame, person):
    """Extract torso region for clothing color analysis"""
    # Get torso keypoints
    neck = person[1]           # Neck
    left_shoulder = person[5]  # Left shoulder  
    right_shoulder = person[2] # Right shoulder
    left_hip = person[12]      # Left hip
    right_hip = person[9]      # Right hip
    
    # Check if we have enough keypoints
    torso_points = []
    if neck[2] > 0.3:
        torso_points.append(neck[:2])
    if left_shoulder[2] > 0.3:
        torso_points.append(left_shoulder[:2])
    if right_shoulder[2] > 0.3:
        torso_points.append(right_shoulder[:2])
    if left_hip[2] > 0.3:
        torso_points.append(left_hip[:2])
    if right_hip[2] > 0.3:
        torso_points.append(right_hip[:2])
    
    if len(torso_points) < 3:
        return None, None
    
    # Create bounding box around torso
    points = np.array(torso_points, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(points)
    
    # Add some padding and ensure within frame bounds
    padding = 20
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(frame.shape[1] - x, w + 2 * padding)
    h = min(frame.shape[0] - y, h + 2 * padding)
    
    if w < 20 or h < 20:  # Too small region
        return None, None
    
    # Extract torso region
    torso_region = frame[y:y+h, x:x+w]
    
    # Create a mask for the torso area (approximate)
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Adjust points relative to the cropped region
    adjusted_points = points - [x, y]
    adjusted_points = np.clip(adjusted_points, [0, 0], [w-1, h-1])
    
    # Create convex hull mask
    if len(adjusted_points) >= 3:
        hull = cv2.convexHull(adjusted_points)
        cv2.fillPoly(mask, [hull], 255)
    else:
        mask.fill(255)  # Use entire region if not enough points
    
    return torso_region, mask

def calculate_color_similarity(color1, color2):
    """Calculate similarity between two RGB colors using perceptual color distance"""
    if color1 is None or color2 is None:
        return 0.0
    
    # Convert RGB to LAB color space for better perceptual comparison
    def rgb_to_lab(rgb):
        # Normalize RGB values
        rgb_normalized = np.array(rgb) / 255.0
        
        # Convert to XYZ
        def gamma_correction(c):
            return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92
        
        rgb_corrected = [gamma_correction(c) for c in rgb_normalized]
        
        # XYZ transformation matrix (sRGB to XYZ)
        xyz = np.dot([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], rgb_corrected)
        
        # Normalize by D65 illuminant
        xyz[0] /= 0.95047
        xyz[1] /= 1.00000
        xyz[2] /= 1.08883
        
        # Convert to LAB
        def f(t):
            return t ** (1/3) if t > 0.008856 else (7.787 * t + 16/116)
        
        fx, fy, fz = [f(c) for c in xyz]
        
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        
        return [L, a, b]
    
    try:
        lab1 = rgb_to_lab(color1)
        lab2 = rgb_to_lab(color2)
        
        # Calculate Delta E (perceptual color difference)
        delta_e = np.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))
        
        # Convert to similarity score (0-1, where 1 is identical)
        # Delta E of 2.3 is considered just noticeable difference
        # Delta E of 10+ is considered very different
        similarity = max(0, 1 - (delta_e / 50))  # Normalize to 0-1 range
        
        return similarity
    except:
        # Fallback to simple RGB distance
        dist = np.linalg.norm(np.array(color1) - np.array(color2))
        max_dist = np.sqrt(3 * 255**2)  # Maximum possible RGB distance
        return max(0, 1 - (dist / max_dist))

def update_person_color(person_id, new_color, confidence=1.0):
    """Update stored color for a person with confidence weighting"""
    if new_color is None:
        return
    
    if person_id not in person_colors:
        person_colors[person_id] = {
            "torso_color": new_color,
            "confidence": confidence,
            "color_history": [new_color]
        }
    else:
        # Weighted average of old and new colors
        old_data = person_colors[person_id]
        old_color = old_data["torso_color"]
        old_confidence = old_data["confidence"]
        
        # Calculate new weighted color
        total_weight = old_confidence + confidence
        new_weighted_color = tuple(
            int((old_color[i] * old_confidence + new_color[i] * confidence) / total_weight)
            for i in range(3)
        )
        
        person_colors[person_id]["torso_color"] = new_weighted_color
        person_colors[person_id]["confidence"] = min(total_weight, 10.0)  # Cap confidence
        
        # Keep color history (last 5 colors)
        if "color_history" not in person_colors[person_id]:
            person_colors[person_id]["color_history"] = []
        person_colors[person_id]["color_history"].append(new_color)
        if len(person_colors[person_id]["color_history"]) > 5:
            person_colors[person_id]["color_history"].pop(0)

def assign_ids_with_color_and_nose(frame, current_people, previous_people, previous_ids, 
                                  nose_threshold=NOSE_DISTANCE_THRESHOLD):
    """
    Assign IDs based on nose position and clothing color
    """
    global person_id_counter
    
    if not previous_people or not current_people:
        # First frame or no previous data
        new_ids = []
        for i, person in enumerate(current_people):
            new_id = f"ID_{person_id_counter}"
            person_id_counter += 1
            new_ids.append(new_id)
            
            # Extract and store clothing color
            torso_region, mask = get_torso_region(frame, person)
            if torso_region is not None:
                color = extract_dominant_color(torso_region, mask)
                update_person_color(new_id, color, confidence=0.5)  # Lower initial confidence
        
        return new_ids
    
    # Calculate similarity matrix
    similarity_matrix = np.zeros((len(current_people), len(previous_people)))
    
    for i, current_person in enumerate(current_people):
        current_nose = current_person[0]  # Nose keypoint
        
        # Extract current person's clothing color
        current_torso_region, current_mask = get_torso_region(frame, current_person)
        current_color = None
        if current_torso_region is not None:
            current_color = extract_dominant_color(current_torso_region, current_mask)
        
        for j, (previous_person, prev_id) in enumerate(zip(previous_people, previous_ids)):
            previous_nose = previous_person[0]
            
            # Calculate position similarity (based on nose)
            position_similarity = 0.0
            if current_nose[2] > 0.3 and previous_nose[2] > 0.3:
                nose_distance = np.linalg.norm(current_nose[:2] - previous_nose[:2])
                if nose_distance <= nose_threshold:
                    position_similarity = max(0, 1 - (nose_distance / nose_threshold))
            
            # Calculate color similarity
            color_similarity = 0.0
            if current_color is not None and prev_id in person_colors:
                stored_color = person_colors[prev_id]["torso_color"]
                color_similarity = calculate_color_similarity(current_color, stored_color)
            
            # Combined similarity score
            total_similarity = (POSITION_WEIGHT * position_similarity + 
                              COLOR_WEIGHT * color_similarity)
            
            similarity_matrix[i, j] = total_similarity
    
    # Hungarian algorithm for optimal assignment (simplified greedy approach)
    new_ids = [None] * len(current_people)
    used_previous = set()
    
    # Sort by highest similarity scores
    assignments = []
    for i in range(len(current_people)):
        for j in range(len(previous_people)):
            if similarity_matrix[i, j] > 0.3:  # Minimum similarity threshold
                assignments.append((similarity_matrix[i, j], i, j))
    
    assignments.sort(reverse=True)  # Highest similarity first
    
    # Assign based on similarity scores
    for similarity, current_idx, previous_idx in assignments:
        if new_ids[current_idx] is None and previous_idx not in used_previous:
            new_ids[current_idx] = previous_ids[previous_idx]
            used_previous.add(previous_idx)
            
            # Update stored color for this person
            current_person = current_people[current_idx]
            torso_region, mask = get_torso_region(frame, current_person)
            if torso_region is not None:
                color = extract_dominant_color(torso_region, mask)
                confidence = min(1.0, similarity * 2)  # Higher similarity = higher confidence
                update_person_color(new_ids[current_idx], color, confidence)
    
    # Assign new IDs to unmatched people
    for i, person_id in enumerate(new_ids):
        if person_id is None:
            new_id = f"ID_{person_id_counter}"
            person_id_counter += 1
            new_ids[i] = new_id
            
            # Store color for new person
            torso_region, mask = get_torso_region(frame, current_people[i])
            if torso_region is not None:
                color = extract_dominant_color(torso_region, mask)
                update_person_color(new_id, color, confidence=0.5)
    
    return new_ids

def visualize_person_colors(frame, keypoints, ids):
    """Add color visualization to the frame for debugging"""
    if keypoints is None:
        return frame
    
    for i, (person, person_id) in enumerate(zip(keypoints, ids)):
        if person_id in person_colors:
            stored_color = person_colors[person_id]["torso_color"]
            confidence = person_colors[person_id]["confidence"]
            
            # Draw color patch near person
            nose = person[0]
            if nose[2] > 0.3:
                x, y = int(nose[0]), int(nose[1])
                
                # Draw color rectangle
                cv2.rectangle(frame, (x - 30, y - 60), (x + 30, y - 30), 
                            (int(stored_color[2]), int(stored_color[1]), int(stored_color[0])), -1)
                
                # Draw border
                cv2.rectangle(frame, (x - 30, y - 60), (x + 30, y - 30), (255, 255, 255), 2)
                
                # Add text
                cv2.putText(frame, f"{person_id}", (x - 25, y - 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"C:{confidence:.1f}", (x - 25, y - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return frame

# Replace your existing assign_ids function call with:
# ids = assign_ids_with_color_and_nose(frame, keypoints, prev_people, prev_ids)

# Add this to your main loop after processing keypoints:
# output_frame = visualize_person_colors(output_frame, keypoints, ids)

print("Enhanced ID assignment with clothing color and nose tracking ready!")
print("Key features:")
print("- Uses nose position for primary tracking")
print("- Analyzes torso region for clothing color")
print("- Combines position and color similarity")
print("- Maintains color history for each person")
print("- Handles lighting changes with confidence weighting")