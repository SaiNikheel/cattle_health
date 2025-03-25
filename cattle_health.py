import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image

# Configurations
THRESHOLD = 20  # Similarity threshold

# Helper function to get subdirectories
def get_subdirectories(base_dir):
    """Returns a list of subdirectories inside the base directory."""
    return [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]

# Helper function to load all images from a subdirectory
def get_all_images(sub_dir):
    """Returns a list of image file paths inside a subdirectory."""
    image_files = [f for f in os.listdir(sub_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return [os.path.join(sub_dir, img) for img in image_files]

# Streamlit UI
st.title("Cattle Nose Pattern Matching (Brute Force)")

# Select a subdirectory
base_dir = "Cattle_data"
subdirs = get_subdirectories(base_dir)

if subdirs:
    selected_subdir = st.selectbox("Select Cattle Subdirectory", subdirs)
    image_paths = get_all_images(os.path.join(base_dir, selected_subdir))

    if image_paths:
        st.write(f"Found {len(image_paths)} images in '{selected_subdir}'")

        # Upload image for comparison
        uploaded_file = st.file_uploader("Upload an image for comparison", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            # Read the uploaded image
            img2 = np.array(Image.open(uploaded_file).convert('RGB'))
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # SIFT for uploaded image
            sift = cv2.SIFT_create()
            keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

            best_match = None
            best_score = 0
            best_img = None
            best_kp = None

            # Iterate over all images in the subdirectory
            for img_path in image_paths:
                img1 = cv2.imread(img_path)
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

                # SIFT for ground truth image
                keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)

                # Brute-Force Matching
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(descriptors1, descriptors2, k=2)

                # Lowe's ratio test
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

                # Compare score with best match
                if len(good_matches) > best_score:
                    best_score = len(good_matches)
                    best_match = img_path
                    best_img = img1
                    best_kp = keypoints1

            # Display the best match results
            if best_match:
                # Draw keypoints and matches
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(sift.detectAndCompute(cv2.cvtColor(best_img, cv2.COLOR_BGR2GRAY), None)[1], descriptors2, k=2)

                # Filter good matches again
                good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

                img_matches = cv2.drawMatches(
                    best_img, best_kp, img2, keypoints2, good_matches, None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )

                # Display the matched image
                st.image(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB), caption=f"Best Match: {os.path.basename(best_match)}", use_container_width=True)

                # Display results
                if best_score > THRESHOLD:
                    st.success(f"✅ Match Found, Score:{best_score}")
                else:
                    st.error(f"❌ No Match")

            else:
                st.warning("No matches found!")

        else:
            st.info("Please upload an image for comparison.")
    else:
        st.warning("No images found in the selected subdirectory.")
else:
    st.warning("No subdirectories found in the 'Cattle_data' folder.")
