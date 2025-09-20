import speech_recognition as sr
import pyaudio
import wave
import threading
import time
import queue
import numpy as np
from typing import Dict, List, Optional, Callable
import logging
from dataclasses import dataclass
from enum import Enum
import re
import json
from datetime import datetime
import webrtcvad
import collections

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ViolationType(Enum):
    UNAUTHORIZED_SPEECH = "unauthorized_speech"
    SUSPICIOUS_KEYWORDS = "suspicious_keywords"
    MULTIPLE_VOICES = "multiple_voices"
    VOICE_CHANGE = "voice_change"
    BACKGROUND_CONVERSATION = "background_conversation"
    PHONE_CALL_DETECTED = "phone_call_detected"

@dataclass
class SpeechViolation:
    violation_type: ViolationType
    confidence: float
    transcript: str
    timestamp: datetime
    audio_segment_path: Optional[str] = None
    speaker_id: Optional[str] = None
    keywords_detected: List[str] = None

class SpeechRecognitionEngine:
    """
    Comprehensive speech recognition system for exam cheating detection
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 chunk_size: int = 1024,
                 channels: int = 1,
                 violation_callback: Optional[Callable] = None):
        
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.violation_callback = violation_callback
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone(sample_rate=sample_rate)
        
        # Initialize VAD (Voice Activity Detection)
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 0-3
        
        # Audio processing
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.audio_thread = None
        
        # Speech analysis
        self.suspicious_keywords = self._load_suspicious_keywords()
        self.speaker_profiles = {}
        self.current_speakers = set()
        
        # Violation tracking
        self.violations = []
        self.speech_segments = []
        
        # Configure recognizer
        self._configure_recognizer()
        
    def _configure_recognizer(self):
        """Configure the speech recognizer settings"""
        # Adjust for ambient noise
        with self.microphone as source:
            logger.info("Calibrating for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            
        # Set recognition parameters
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recognizer.operation_timeout = None
        
    def _load_suspicious_keywords(self) -> Dict[str, List[str]]:
        """Load suspicious keywords that might indicate cheating"""
        return {
            "help_requests": [
                "help", "answer", "tell me", "what is", "how do", "can you",
                "please help", "i need help", "stuck", "don't know"
            ],
            "communication": [
                "phone", "call", "text", "message", "chat", "talking to",
                "someone", "person", "friend", "classmate"
            ],
            "cheating_indicators": [
                "google", "search", "look up", "find answer", "copy",
                "paste", "cheat", "share", "send", "website", "internet"
            ],
            "coordination": [
                "ready", "start", "begin", "finish", "done", "next question",
                "which one", "number", "option", "choice"
            ]
        }
    
    def start_monitoring(self):
        """Start continuous audio monitoring"""
        if self.is_listening:
            logger.warning("Already listening")
            return
            
        self.is_listening = True
        self.audio_thread = threading.Thread(target=self._audio_processing_loop)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        
        logger.info("Speech monitoring started")
    
    def stop_monitoring(self):
        """Stop audio monitoring"""
        self.is_listening = False
        if self.audio_thread:
            self.audio_thread.join()
        logger.info("Speech monitoring stopped")
    
    def _audio_processing_loop(self):
        """Main audio processing loop"""
        try:
            with self.microphone as source:
                while self.is_listening:
                    try:
                        # Listen for audio with timeout
                        audio_data = self.recognizer.listen(
                            source, 
                            timeout=1, 
                            phrase_time_limit=5
                        )
                        
                        # Process audio in separate thread
                        processing_thread = threading.Thread(
                            target=self._process_audio_segment,
                            args=(audio_data,)
                        )
                        processing_thread.daemon = True
                        processing_thread.start()
                        
                    except sr.WaitTimeoutError:
                        # No speech detected, continue
                        continue
                    except Exception as e:
                        logger.error(f"Audio processing error: {e}")
                        
        except Exception as e:
            logger.error(f"Audio loop error: {e}")
    
    def _process_audio_segment(self, audio_data):
        """Process individual audio segment"""
        try:
            # Convert to numpy array for analysis
            audio_array = np.frombuffer(audio_data.get_raw_data(), dtype=np.int16)
            
            # Voice Activity Detection
            if not self._is_speech(audio_array):
                return
            
            # Speech-to-text conversion
            transcript = self._transcribe_audio(audio_data)
            if not transcript:
                return
                
            # Analyze the transcript
            violations = self._analyze_speech(transcript, audio_data)
            
            # Store speech segment
            self._store_speech_segment(transcript, audio_data)
            
            # Process violations
            for violation in violations:
                self._handle_violation(violation)
                
        except Exception as e:
            logger.error(f"Error processing audio segment: {e}")
    
    def _is_speech(self, audio_array: np.ndarray) -> bool:
        """Use VAD to determine if audio contains speech"""
        try:
            # Convert to bytes for VAD
            audio_bytes = audio_array.astype(np.int16).tobytes()
            
            # VAD requires specific frame sizes (10, 20, or 30 ms)
            frame_duration = 30  # ms
            frame_size = int(self.sample_rate * frame_duration / 1000)
            
            # Check if we have enough samples
            if len(audio_array) < frame_size:
                return False
            
            # Take first frame for VAD check
            frame = audio_bytes[:frame_size * 2]  # 2 bytes per sample
            
            return self.vad.is_speech(frame, self.sample_rate)
            
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return True  # Default to processing if VAD fails
    
    def _transcribe_audio(self, audio_data) -> Optional[str]:
        """Convert speech to text"""
        try:
            # Try Google Speech Recognition first
            try:
                transcript = self.recognizer.recognize_google(
                    audio_data, 
                    language='en-US'
                )
                return transcript.lower().strip()
            except sr.RequestError:
                # Fallback to offline recognition
                try:
                    transcript = self.recognizer.recognize_sphinx(audio_data)
                    return transcript.lower().strip()
                except sr.RequestError:
                    logger.warning("No speech recognition service available")
                    return None
                    
        except sr.UnknownValueError:
            # Speech was unintelligible
            return None
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None
    
    def _analyze_speech(self, transcript: str, audio_data) -> List[SpeechViolation]:
        """Analyze transcript for suspicious content"""
        violations = []
        
        # Check for suspicious keywords
        keyword_violation = self._check_suspicious_keywords(transcript)
        if keyword_violation:
            violations.append(keyword_violation)
        
        # Check for multiple speakers
        speaker_violation = self._check_multiple_speakers(transcript, audio_data)
        if speaker_violation:
            violations.append(speaker_violation)
        
        # Check for phone call patterns
        phone_violation = self._check_phone_call_patterns(transcript)
        if phone_violation:
            violations.append(phone_violation)
        
        # Check for unauthorized speech during quiet periods
        if self._is_unauthorized_speech_time():
            violations.append(SpeechViolation(
                violation_type=ViolationType.UNAUTHORIZED_SPEECH,
                confidence=0.8,
                transcript=transcript,
                timestamp=datetime.now()
            ))
        
        return violations
    
    def _check_suspicious_keywords(self, transcript: str) -> Optional[SpeechViolation]:
        """Check for suspicious keywords in transcript"""
        detected_keywords = []
        max_confidence = 0.0
        
        for category, keywords in self.suspicious_keywords.items():
            for keyword in keywords:
                if keyword in transcript:
                    detected_keywords.append(keyword)
                    # Higher confidence for more specific cheating terms
                    if category == "cheating_indicators":
                        max_confidence = max(max_confidence, 0.9)
                    elif category == "communication":
                        max_confidence = max(max_confidence, 0.8)
                    else:
                        max_confidence = max(max_confidence, 0.6)
        
        if detected_keywords:
            return SpeechViolation(
                violation_type=ViolationType.SUSPICIOUS_KEYWORDS,
                confidence=max_confidence,
                transcript=transcript,
                timestamp=datetime.now(),
                keywords_detected=detected_keywords
            )
        
        return None
    
    def _check_multiple_speakers(self, transcript: str, audio_data) -> Optional[SpeechViolation]:
        """Check for multiple speakers in audio"""
        # Simple heuristic: look for conversation patterns
        conversation_patterns = [
            r'\b(yes|no|okay|sure)\b.*\b(what|how|when|where|why)\b',
            r'\b(hello|hi|hey)\b',
            r'\b(thanks|thank you|please)\b',
            r'\?\s*\w+',  # Question followed by response
        ]
        
        for pattern in conversation_patterns:
            if re.search(pattern, transcript, re.IGNORECASE):
                return SpeechViolation(
                    violation_type=ViolationType.MULTIPLE_VOICES,
                    confidence=0.7,
                    transcript=transcript,
                    timestamp=datetime.now()
                )
        
        return None
    
    def _check_phone_call_patterns(self, transcript: str) -> Optional[SpeechViolation]:
        """Check for phone call indicators"""
        phone_patterns = [
            r'\b(hello|hi)\b.*\b(calling|call)\b',
            r'\bcan you hear me\b',
            r'\bhang up\b',
            r'\bphone|mobile|cell\b',
            r'\bdial|number\b'
        ]
        
        for pattern in phone_patterns:
            if re.search(pattern, transcript, re.IGNORECASE):
                return SpeechViolation(
                    violation_type=ViolationType.PHONE_CALL_DETECTED,
                    confidence=0.8,
                    transcript=transcript,
                    timestamp=datetime.now()
                )
        
        return None
    
    def _is_unauthorized_speech_time(self) -> bool:
        """Check if speech is occurring during a quiet period"""
        # This would integrate with your exam state management
        # For now, return False as a placeholder
        return False
    
    def _store_speech_segment(self, transcript: str, audio_data):
        """Store speech segment for evidence"""
        segment = {
            'timestamp': datetime.now().isoformat(),
            'transcript': transcript,
            'duration': len(audio_data.get_raw_data()) / (self.sample_rate * 2),  # seconds
        }
        self.speech_segments.append(segment)
        
        # Keep only recent segments (last 100)
        if len(self.speech_segments) > 100:
            self.speech_segments = self.speech_segments[-100:]
    
    def _handle_violation(self, violation: SpeechViolation):
        """Handle detected violation"""
        self.violations.append(violation)
        
        logger.warning(f"Speech violation detected: {violation.violation_type.value}")
        logger.info(f"Transcript: {violation.transcript}")
        logger.info(f"Confidence: {violation.confidence:.2f}")
        
        # Call violation callback if provided
        if self.violation_callback:
            self.violation_callback(violation)
    
    def get_violations_summary(self) -> Dict:
        """Get summary of all violations"""
        violation_counts = {}
        for violation in self.violations:
            v_type = violation.violation_type.value
            violation_counts[v_type] = violation_counts.get(v_type, 0) + 1
        
        return {
            'total_violations': len(self.violations),
            'violation_breakdown': violation_counts,
            'recent_violations': [
                {
                    'type': v.violation_type.value,
                    'confidence': v.confidence,
                    'transcript': v.transcript,
                    'timestamp': v.timestamp.isoformat(),
                    'keywords': v.keywords_detected
                }
                for v in self.violations[-10:]  # Last 10 violations
            ]
        }
    
    def save_audio_evidence(self, violation: SpeechViolation, audio_data, filename: str):
        """Save audio evidence for violation"""
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data.get_raw_data())
            
            violation.audio_segment_path = filename
            logger.info(f"Audio evidence saved: {filename}")
            
        except Exception as e:
            logger.error(f"Error saving audio evidence: {e}")


class RealTimeSpeechMonitor:
    """
    Real-time speech monitoring system for exam sessions
    """
    
    def __init__(self, session_id: str, student_id: str):
        self.session_id = session_id
        self.student_id = student_id
        self.speech_engine = None
        self.monitoring_active = False
        
    def start_session(self):
        """Start monitoring for exam session"""
        def violation_handler(violation: SpeechViolation):
            self._process_violation(violation)
        
        self.speech_engine = SpeechRecognitionEngine(
            violation_callback=violation_handler
        )
        
        self.speech_engine.start_monitoring()
        self.monitoring_active = True
        
        logger.info(f"Speech monitoring started for session {self.session_id}")
    
    def stop_session(self):
        """Stop monitoring and generate report"""
        if self.speech_engine:
            self.speech_engine.stop_monitoring()
        
        self.monitoring_active = False
        
        # Generate final report
        report = self._generate_session_report()
        logger.info(f"Speech monitoring stopped for session {self.session_id}")
        
        return report
    
    def _process_violation(self, violation: SpeechViolation):
        """Process detected violation"""
        # Here you would integrate with your main violation system
        violation_data = {
            'session_id': self.session_id,
            'student_id': self.student_id,
            'violation_type': violation.violation_type.value,
            'confidence': violation.confidence,
            'transcript': violation.transcript,
            'timestamp': violation.timestamp.isoformat(),
            'evidence_type': 'speech'
        }
        
        # Save to database or send to violation handler
        logger.warning(f"Processing speech violation: {violation_data}")
        
        # Save audio evidence if needed
        if hasattr(violation, 'audio_data'):
            filename = f"evidence/{self.session_id}_{violation.timestamp.strftime('%Y%m%d_%H%M%S')}.wav"
            self.speech_engine.save_audio_evidence(violation, violation.audio_data, filename)
    
    def _generate_session_report(self) -> Dict:
        """Generate final session report"""
        if not self.speech_engine:
            return {}
        
        violations_summary = self.speech_engine.get_violations_summary()
        
        report = {
            'session_id': self.session_id,
            'student_id': self.student_id,
            'monitoring_duration': 'calculated_duration',  # Calculate actual duration
            'speech_violations': violations_summary,
            'generated_at': datetime.now().isoformat()
        }
        
        return report


# Example usage and integration
if __name__ == "__main__":
    def handle_speech_violation(violation: SpeechViolation):
        print(f"🚨 VIOLATION DETECTED: {violation.violation_type.value}")
        print(f"📝 Transcript: {violation.transcript}")
        print(f"🎯 Confidence: {violation.confidence:.2f}")
        print("-" * 50)
    
    # Initialize speech recognition system
    speech_system = SpeechRecognitionEngine(violation_callback=handle_speech_violation)
    
    print("Starting speech monitoring...")
    print("Speak to test the system. Press Ctrl+C to stop.")
    
    try:
        speech_system.start_monitoring()
        
        # Keep monitoring until interrupted
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping speech monitoring...")
        speech_system.stop_monitoring()
        
        # Print summary
        summary = speech_system.get_violations_summary()
        print("\n📊 MONITORING SUMMARY:")
        print(json.dumps(summary, indent=2))