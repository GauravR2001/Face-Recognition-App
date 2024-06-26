import os
import pickle
import streamlit as st
from deepface import DeepFace
import numpy as np
from PIL import Image, ImageDraw
import io


db_path = os.path.join(os.getcwd(), 'DB')
output_file = os.path.join(os.getcwd(), 'Embed', 'embeddings2.pkl')



if not os.path.exists(db_path):
    os.makedirs(db_path)


# Function to create embeddings pickle
def create_embeddings_pickle(db_path, output_file):
    embeddings_dict = {}
    for person_name in os.listdir(db_path):
        person_dir = os.path.join(db_path, person_name)
        if not os.path.isdir(person_dir):
            continue
        person_embeddings = []
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            try:
                embedding = DeepFace.represent(img_path=image_path, enforce_detection=False)[0]["embedding"]
                person_embeddings.append(embedding)
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue
        if person_embeddings:
            embeddings_dict[person_name] = person_embeddings
            print(f"Processed {len(person_embeddings)} images for {person_name}")
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings_dict, f)
    print(f"Embeddings saved to {output_file}")

# Function to represent an image as an embedding
def represent(image):
    embedding = DeepFace.represent(img_path=image, enforce_detection=False)[0]['embedding']
    return np.array(embedding)

# Function to calculate Euclidean distance
def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# Function to prepare the embeddings dictionary
def data(embedding_dict):
    return embedding_dict  # Each person's entry will contain multiple embeddings

# Function to compare the target image with the embeddings
def comparison(target_embedding, embed, threshold):
    best_match = "Not recognized"
    best_distance = None
    
    for person_name, person_embeddings in embed.items():
        for person_embedding in person_embeddings:
            dist = distance(target_embedding, person_embedding)
            if dist < threshold:
                if best_distance is None or dist < best_distance:
                    best_distance = dist
                    best_match = person_name
    
    return best_match, best_distance

# Function to load embeddings
def load_embeddings_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Function to create a new user directory
def create_user_directory(db_path, new_user_name):
    if new_user_name:
        new_user_dir = os.path.join(db_path, new_user_name)
        try:
            os.makedirs(new_user_dir, exist_ok=True)
            return new_user_dir  # Return the path to the new user directory
        except Exception as e:
            print(f"Error creating directory for {new_user_name}: {e}")
            return None
    return None

# Streamlit interface
def main():
    st.set_page_config(page_title="Face Recognition with Capture", page_icon=":camera:", layout="wide")

    st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .main {
        background-color: black;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }
    h1 {
        color: #4CAF50;
        font-family: 'Arial';
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 15px 20px;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .result-success {
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #000000;
        background-color: #FFFF00;
        padding: 10px;
        border-radius: 5px;
        margin-top: 20px;
    }
    .result-error {
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #ffffff;
        background-color: #f44336;
        padding: 10px;
        border-radius: 5px;
        margin-top: 20px;
    }
    .result-info {
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #ffffff;
        background-color: #008CBA;
        padding: 10px;
        border-radius: 5px;
        margin-top: 20px;
    }
    .video-frame {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .stButtonUnique>button {
        width: 100%;
        background-color: #008CBA;
        color: white;
        padding: 15px 20px;
        border-radius: 5px;
        border: none;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Face Recognition with Capture :camera:")
    
    # Main menu selection
    menu_choice = st.sidebar.radio("Select Action", ("Attendance", "New User"))

    if menu_choice == "Attendance":
        st.write("Click 'Capture Image' to capture an image and recognize faces in real-time.")

        with st.expander("Instructions"):
            st.write("""
            1. Click 'Capture Image' to capture an image from your webcam.
            2. Results will be displayed below.
            """)

        st.sidebar.title("Settings")
        threshold = st.sidebar.slider("Threshold for Recognition", min_value=0.5, max_value=2.0, value=1.2, step=0.1)

        # db_path = r"C:\Users\DH 1025 TX\Documents\Jupyter\DeepFace\DB"  # Update this path as needed
        # output_file = r"C:\Users\DH 1025 TX\Documents\Jupyter\DeepFace\Embed\embeddings2.pkl"
        
        if st.sidebar.button('Update Embeddings'):
            create_embeddings_pickle(db_path, output_file)
            st.sidebar.success("Embeddings updated successfully.")

        dataset = load_embeddings_pickle(output_file)
        newdata = data(dataset)

        placeholder = st.empty()
        result_placeholder = st.empty()

        # Use Streamlit's built-in camera input
        captured_image = st.camera_input("Capture Image")

        if captured_image:
            # Convert the image to PIL format
            image = Image.open(io.BytesIO(captured_image.read()))

            try:
                # Convert image to a format suitable for DeepFace
                image_np = np.array(image)

                # Extract faces and perform anti-spoofing
                faces = DeepFace.extract_faces(img_path=image_np, enforce_detection=False, anti_spoofing=True)

                if not faces:
                    result_placeholder.markdown('<div class="result-error">No face detected.</div>', unsafe_allow_html=True)
                else:
                    results = []
                    for face_obj in faces:
                        if not face_obj['is_real']:
                            results.append('<div class="result-error">Spoof detected! This appears to be a fake image.</div>')
                            continue

                        x, y, w, h, _, _ = face_obj['facial_area'].values()
                        face_frame = image_np[y:y+h, x:x+w]

                        # Compare with embeddings
                        face_embedding = represent(face_frame)
                        result_text, result_distance = comparison(face_embedding, newdata, threshold)

                        if result_text != 'Not recognized':
                            results.append(f'<div class="result-success"> Welcome {result_text}, distance: {result_distance} </div>')
                        else:
                            results.append('<div class="result-error">Not recognized</div>')

                        # Draw rectangle and result on the image
                        draw = ImageDraw.Draw(image)
                        color = "green" if result_text != 'Not recognized' else "red"
                        draw.rectangle([(x, y), (x+w, y+h)], outline=color, width=3)
                        draw.text((x, y-10), result_text, fill=color)

                    # Display captured frame with results
                    placeholder.image(image, caption="Captured Frame", use_column_width=True)

                    # Display results for each detected face
                    for result in results:
                        result_placeholder.markdown(result, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error in extracting faces: {e}")

    elif menu_choice == "New User":
        st.write("Add a new user: enter their name and capture images.")

        with st.expander("Instructions"):
            st.write("""
            1. Enter the new user's name in the sidebar.
            2. Click 'Create User Directory' to create a directory for the new user.
            3. Click 'Add Pictures' to capture images for the new user.
            """)

        st.sidebar.title("New User Setup")
        new_user_name = st.sidebar.text_input("Enter new user's name:")
        #db_path = r"C:\Users\DH 1025 TX\Documents\Jupyter\DeepFace\DB"  # Update this path as needed

        # Initialize session state
        if 'new_user_dir' not in st.session_state:
            st.session_state.new_user_dir = None

        if new_user_name:
            if st.sidebar.button('Create User Directory'):
                new_user_dir = create_user_directory(db_path, new_user_name)
                if new_user_dir:
                    st.session_state.new_user_dir = new_user_dir  # Store new user directory in session state
                    st.sidebar.success(f"Directory created for {new_user_name}")

        # Update embeddings button
        if st.sidebar.button('Update Embeddings'):
            create_embeddings_pickle(db_path, output_file)
            st.sidebar.success("Embeddings updated successfully.")

        if st.session_state.new_user_dir:
            st.sidebar.button('Add Pictures')   
            captured_image = st.camera_input("Capture Image")
            if captured_image:
                image = Image.open(io.BytesIO(captured_image.read()))
                try:
                    image_np = np.array(image)
                    image_path = os.path.join(st.session_state.new_user_dir, f"image_{len(os.listdir(st.session_state.new_user_dir)) + 1}.jpg")
                    image.save(image_path)
                    st.success("Image added successfully.")
                except Exception as e:
                    st.error(f"Error in adding image: {e}")
            else:
                st.warning("Please capture an image.")
        else:
            st.sidebar.warning("Please enter a name for the new user.")

if __name__ == "__main__":
    main()
