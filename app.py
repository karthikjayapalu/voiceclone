import streamlit as st
import soundfile as sf
import numpy as np
import os
import tempfile
import torch
import io
from voxcpm import VoxCPM
import base64
import kagglehub
import glob
from audio_recorder_streamlit import audio_recorder

# Page configuration
st.set_page_config(
    page_title="Voice Clone & TTS",
    page_icon="🎙️",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #2a5298;
        color: white;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #1e3c72;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        color: #155724;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        color: #856404;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h1>🎙️ Voice Clone & Text-to-Speech</h1><p>Clone any voice and generate speech using AI</p></div>', unsafe_allow_html=True)

# Initialize session state for model and audio
if 'model' not in st.session_state:
    st.session_state.model = None
if 'reference_audio' not in st.session_state:
    st.session_state.reference_audio = None
if 'reference_text' not in st.session_state:
    st.session_state.reference_text = ""
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

@st.cache_data
def download_charlie_kirk_dataset():
    """Download Charlie Kirk dataset from Kaggle"""
    try:
        with st.spinner("Downloading Charlie Kirk dataset from Kaggle..."):
            path = kagglehub.dataset_download("bwandowando/charlie-kirk-twitter-dataset")
            return path
    except Exception as e:
        st.warning(f"Could not download dataset: {str(e)}")
        return None

def get_sample_audio_from_dataset(dataset_path):
    """Find a suitable audio file from the dataset"""
    if not dataset_path or not os.path.exists(dataset_path):
        return None, None
    
    # Look for audio/video files (common formats)
    audio_extensions = ['*.wav', '*.mp3', '*.m4a', '*.flac', '*.mp4']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(dataset_path, '**', ext), recursive=True))
    
    if audio_files:
        # Return the first audio file found
        return audio_files[0], "Sample from Charlie Kirk dataset"
    
    return None, None

@st.cache_resource
def load_model():
    """Load the VoxCPM model (cached)"""
    try:
        # Check for MPS (Apple Silicon) or CUDA
        if torch.backends.mps.is_available():
            device = "mps"
            st.info("🍎 Apple Silicon MPS detected - using Metal acceleration")
        elif torch.cuda.is_available():
            device = "cuda"
            st.info("🚀 CUDA detected - using GPU acceleration")
        else:
            device = "cpu"
            st.warning("⚠️ No GPU detected - using CPU (this will be slow)")
        
        with st.spinner("Loading VoxCPM model... This may take a few minutes on first run."):
            model = VoxCPM.from_pretrained("openbmb/VoxCPM1.5")
            return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("💡 Tip: Make sure you have sufficient disk space and internet connection")
        return None

def generate_speech(text, reference_audio_path, reference_text, cfg_value=2.0, timesteps=10):
    """Generate speech using the model"""
    try:
        # Clear MPS cache if needed
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Check if reference audio exists
        if not os.path.exists(reference_audio_path):
            st.error(f"Reference audio file not found: {reference_audio_path}")
            return None
        
        generated_wav = st.session_state.model.generate(
            text=text,
            prompt_wav_path=reference_audio_path,
            prompt_text=reference_text,
            cfg_value=cfg_value,
            inference_timesteps=timesteps,
            denoise=True,
        )
        return generated_wav
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        return None

def get_audio_player(audio_data, sample_rate):
    """Create an HTML audio player for the generated audio"""
    # Save to bytes buffer
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format='wav')
    buffer.seek(0)
    
    # Convert to base64 for HTML playback
    audio_base64 = base64.b64encode(buffer.read()).decode()
    audio_html = f"""
        <audio controls style="width: 100%;">
            <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
            Your browser does not support the audio element.
        </audio>
    """
    return audio_html

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = ['setuptools', 'voxcpm', 'torch', 'soundfile', 'streamlit']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def main():
    # Check dependencies first
    missing_packages = check_dependencies()
    if missing_packages:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning(f"⚠️ Missing required packages: {', '.join(missing_packages)}")
        st.code("pip install " + " ".join(missing_packages))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model loading section
        st.subheader("1. Load Model")
        
        # Show device info
        if torch.cuda.is_available():
            st.success("✅ CUDA available")
        elif torch.backends.mps.is_available():
            st.success("✅ MPS available (Apple Silicon)")
        else:
            st.warning("⚠️ Using CPU (slow)")
        
        if st.button("🔄 Load VoxCPM Model", use_container_width=True):
            with st.spinner("Loading model... This may take a few minutes..."):
                st.session_state.model = load_model()
                if st.session_state.model:
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded successfully!")
                    st.rerun()
        
        if not st.session_state.model_loaded:
            st.warning("⚠️ Please load the model first")
            st.stop()
        
        # Voice sample configuration
        st.subheader("2. Configure Voice Sample")
        
        # Option to upload custom sample or record
        upload_option = st.radio(
            "Choose voice sample source:",
            ["Record your voice", "Upload audio file", "Download Charlie Kirk (Kaggle)"]
        )
        
        if upload_option == "Record your voice":
            st.info("🎤 Click the microphone button below to record your voice sample")
            st.markdown("**Tips for best results:**")
            st.markdown("- Record 5-10 seconds of clear speech")
            st.markdown("- Speak naturally in a quiet environment")
            st.markdown("- Avoid background noise")
            
            # Audio recorder
            audio_bytes = audio_recorder()
            
            if audio_bytes:
                # Save recorded audio
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(audio_bytes)
                    st.session_state.reference_audio = tmp_file.name
                
                st.success("✅ Voice recorded successfully!")
                st.audio(audio_bytes, format='audio/wav')
                
                st.session_state.reference_text = st.text_area(
                    "Enter what you said in the recording:",
                    value=st.session_state.reference_text,
                    help="Transcript helps improve voice cloning accuracy",
                    height=100,
                    placeholder="Type the exact words you spoke in the recording..."
                )
        
        elif upload_option == "Upload audio file":
            uploaded_file = st.file_uploader(
                "Upload voice sample (WAV/MP3/MP4 format)",
                type=['wav', 'mp3', 'mp4']
            )
            if uploaded_file is not None:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    st.session_state.reference_audio = tmp_file.name
                
                st.session_state.reference_text = st.text_area(
                    "Enter the exact transcript of the audio:",
                    value=st.session_state.reference_text,
                    help="This helps the model match the voice more accurately",
                    height=100
                )
                
                # Play uploaded audio
                audio_bytes = uploaded_file.getvalue()
                st.audio(audio_bytes, format='audio/wav')
        
        else:  # Download Charlie Kirk from Kaggle
            if st.button("📥 Download Charlie Kirk Dataset", use_container_width=True):
                dataset_path = download_charlie_kirk_dataset()
                if dataset_path:
                    st.info(f"📁 Dataset downloaded to: {dataset_path}")
                    audio_file, transcript = get_sample_audio_from_dataset(dataset_path)
                    if audio_file:
                        st.session_state.reference_audio = audio_file
                        st.session_state.reference_text = transcript or "Sample from Charlie Kirk"
                        st.success(f"✅ Found audio: {os.path.basename(audio_file)}")
                    else:
                        st.warning("No audio files found in dataset. Please upload a custom sample.")
            
            if st.session_state.reference_audio and os.path.exists(st.session_state.reference_audio):
                st.info(f"📁 Using: {os.path.basename(st.session_state.reference_audio)}")
                # Allow editing transcript
                st.session_state.reference_text = st.text_area(
                    "Transcript (optional - edit if needed):",
                    value=st.session_state.reference_text,
                    help="Provide the transcript of what's said in the audio",
                    height=100
                )
                # Play the audio
                try:
                    with open(st.session_state.reference_audio, 'rb') as f:
                        st.audio(f.read(), format='audio/wav')
                except:
                    pass
        
        # Advanced parameters
        st.subheader("3. Advanced Parameters")
        cfg_value = st.slider(
            "CFG Value (style adherence)",
            min_value=1.0,
            max_value=3.0,
            value=2.0,
            step=0.1,
            help="Higher values follow the reference voice more closely"
        )
        
        timesteps = st.slider(
            "Inference Timesteps",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
            help="Higher values = better quality but slower generation"
        )

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Enter Text to Convert")
        text_input = st.text_area(
            "Type or paste the text you want to convert to speech:",
            height=150,
            placeholder="Enter any text here... The AI will speak it in the cloned voice.",
            key="text_input"
        )
        
        # Character count
        char_count = len(text_input)
        st.caption(f"Characters: {char_count}")
        
        # Generate button
        generate_button = st.button(
            "🎙️ Generate Speech",
            type="primary",
            use_container_width=True,
            disabled=not (text_input and st.session_state.reference_audio and st.session_state.reference_text)
        )
    
    with col2:
        st.header("ℹ️ Instructions")
        st.info("""
        1. Load the VoxCPM model
        2. Record/upload voice sample
        3. Provide transcript (optional)
        4. Enter text to generate
        5. Click 'Generate Speech'
        
        **Tip:** Longer, clearer voice samples produce better clones.
        """)

    # Generation and playback
    if generate_button:
        if st.session_state.model and st.session_state.reference_audio:
            with st.spinner("🎤 Generating speech... This may take a moment."):
                # Show progress info
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Processing audio...")
                progress_bar.progress(25)
                
                # Generate speech
                generated_audio = generate_speech(
                    text=text_input,
                    reference_audio_path=st.session_state.reference_audio,
                    reference_text=st.session_state.reference_text,
                    cfg_value=cfg_value,
                    timesteps=timesteps
                )
                
                progress_bar.progress(75)
                status_text.text("Finalizing...")
                
                if generated_audio is not None:
                    st.session_state.generated_audio = generated_audio
                    st.session_state.sample_rate = st.session_state.model.tts_model.sample_rate
                    
                    progress_bar.progress(100)
                    status_text.text("Complete!")
                    
                    st.markdown('<div class="success-box">✅ Speech generated successfully!</div>', unsafe_allow_html=True)
                else:
                    st.error("Failed to generate speech")
                
                # Clear progress indicators after 2 seconds
                import time
                time.sleep(2)
                progress_bar.empty()
                status_text.empty()
    
    # Display generated audio if available
    if hasattr(st.session_state, 'generated_audio') and st.session_state.generated_audio is not None:
        st.header("🎧 Generated Audio")
        
        # Create audio player
        audio_html = get_audio_player(
            st.session_state.generated_audio, 
            st.session_state.sample_rate
        )
        st.markdown(audio_html, unsafe_allow_html=True)
        
        # Download button
        buffer = io.BytesIO()
        sf.write(
            buffer, 
            st.session_state.generated_audio, 
            st.session_state.sample_rate, 
            format='wav'
        )
        buffer.seek(0)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="💾 Download Audio (WAV)",
                data=buffer,
                file_name="cloned_voice_speech.wav",
                mime="audio/wav",
                use_container_width=True
            )
        
        with col2:
            if st.button("🔄 Clear Audio", use_container_width=True):
                del st.session_state.generated_audio
                st.rerun()
        
        # Audio info
        duration = len(st.session_state.generated_audio) / st.session_state.sample_rate
        st.caption(f"Duration: {duration:.2f} seconds | Sample Rate: {st.session_state.sample_rate} Hz")

if __name__ == "__main__":
    main()