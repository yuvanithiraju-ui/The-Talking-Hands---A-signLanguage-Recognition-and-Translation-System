# talking_hands_translator.py
# Full Talking Hands app + Multilanguage translate & speak (gTTS) + Tkinter UI

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')   # ensure UTF-8 printing on Windows consoles

import math
import base64
import tempfile
import threading
import traceback
import time

import cv2
import numpy as np
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

import pyttsx3
import enchant

# Translation & TTS
from googletrans import Translator, LANGUAGES
from gtts import gTTS
import playsound

# ------------------- CONFIG -------------------
OFFSET = 29
IMG_SIZE = 400
MODEL_PATH = "cnn8grps_rad1_model.h5"
WHITE_IMG_PATHS = ["white.jpg", os.path.join(os.path.dirname(os.path.abspath(__file__)), "white.jpg")]

# ------------------- HELPERS -------------------
def safe_load_white():
    """Try to load white.jpg else create white canvas"""
    for p in WHITE_IMG_PATHS:
        if os.path.exists(p):
            try:
                img = cv2.imread(p)
                if img is not None and img.size > 0:
                    # Resize to 400x400 if not already
                    try:
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    except:
                        pass
                    return img
            except:
                pass
    # fallback white image 400x400
    return np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255

# ------------------- APPLICATION -------------------
class Application:
    def __init__(self):
        # Model, detectors, translator, voice engines
        self.model = None
        self.load_model()

        # Improved hand detection parameters for better accuracy
        self.hd = HandDetector(maxHands=1, detectionCon=0.7, minTrackCon=0.7)
        self.hd2 = HandDetector(maxHands=1, detectionCon=0.7, minTrackCon=0.7)

        # Initialize dictionary for word suggestions
        self.dict_en = None
        try:
            print("Initializing dictionary for word suggestions...")
            self.dict_en = enchant.Dict("en-US")
            print("Dictionary initialized successfully")
        except Exception as e:
            print("pyenchant dictionary not available:", e)
            print("Creating simple custom dictionary for suggestions...")
            # Create a simple custom dictionary with common English words
            self.common_words = ["the", "be", "to", "of", "and", "a", "in", "that", "have", "I", 
                               "it", "for", "not", "on", "with", "he", "as", "you", "do", "at", 
                               "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", 
                               "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", 
                               "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", 
                               "when", "make", "can", "like", "time", "no", "just", "him", "know", "take", 
                               "people", "into", "year", "your", "good", "some", "could", "them", "see", "other", 
                               "than", "then", "now", "look", "only", "come", "its", "over", "think", "also", 
                               "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", 
                               "even", "new", "want", "because", "any", "these", "give", "day", "most", "us","Good","Morning"]
            
            # Create a custom dictionary class with suggest method
            class CustomDict:
                def __init__(self, words):
                    self.words = words
                
                def suggest(self, word):
                    # Simple suggestion algorithm - find words that start with the same letters
                    word = word.lower()
                    suggestions = [w for w in self.words if w.startswith(word) and w != word]
                    # Add some words that are similar (edit distance)
                    for w in self.words:
                        if w.lower() != word and len(w) > 2 and (word in w.lower() or w.lower() in word):
                            if w not in suggestions:
                                suggestions.append(w)
                    return suggestions[:4]  # Return up to 4 suggestions
            
            self.dict_en = CustomDict(self.common_words)
            print("Custom dictionary created successfully")

        # Initialize translator with more robust error handling
        self.translator = None
        try:
            print("Initializing translator...")
            self.translator = Translator(service_urls=['translate.google.com'])
            # Test the translator with a simple translation
            test = self.translator.translate("Hello", dest="en")
            if not test or not test.text:
                print("Warning: Translator initialization test failed, using fallback")
                self.translator = None
            else:
                print("Translator initialized successfully")
        except Exception as e:
            print("Translator initialization error:", e)
            print("Translation functionality will be limited")
            self.translator = None

        # pyttsx3 fallback engine (English)
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty("rate", 120)
        voices = self.tts_engine.getProperty("voices")
        # Try to pick a female voice if available in engine voices
        if voices:
            # pick a voice with 'female' in name or the second entry if available
            chosen = None
            for v in voices:
                try:
                    if 'female' in v.name.lower():
                        chosen = v.id
                        break
                except:
                    continue
            if not chosen and len(voices) > 1:
                chosen = voices[1].id
            if chosen:
                try:
                    self.tts_engine.setProperty("voice", chosen)
                except:
                    pass

        # UI / state variables
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None

        self.ct = {c: 0 for c in ascii_uppercase}
        self.ct['blank'] = 0

        self.blank_flag = 0
        self.space_flag = False
        self.next_flag = True
        self.prev_char = ""
        self.count = -1
        self.ten_prev_char = [" "]*10

        self.pts = None
        self.current_symbol = " "
        self.str = " "
        self.word = " "
        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "

        # translation state
        self.translated_sentence = None
        self.selected_lang_code = "en"  # default english
        self.lang_map = {
            "English":"en", "Tamil":"ta", "Hindi":"hi", "Telugu":"te", "Kannada":"kn", "Malayalam":"ml",
            "Gujarati":"gu", "Bengali":"bn", "Marathi":"mr", "Punjabi":"pa", "Urdu":"ur", "French":"fr",
            "Spanish":"es", "German":"de", "Italian":"it", "Japanese":"ja", "Chinese (Simplified)":"zh-cn",
            "Korean":"ko", "Russian":"ru", "Arabic":"ar"
        }

        # build UI
        self.build_ui()

        # loop
        self.root.after(1, self.video_loop)

    def load_model(self):
        try:
            print("Loading model:", MODEL_PATH)
            self.model = load_model(MODEL_PATH)
            print("Model loaded.")
        except Exception as e:
            print("Failed to load model:", e)
            self.model = None

    def build_ui(self):
        self.root = tk.Tk()
        self.root.title("The Talking Hands - Multilingual")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        # adjust size to fit dropdown at bottom-right
        self.root.geometry("1400x780")

        # left video panel
        self.panel = tk.Label(self.root)
        self.panel.place(x=100, y=3, width=480, height=640)

        # processed skeleton preview
        self.panel2 = tk.Label(self.root)
        self.panel2.place(x=700, y=115, width=400, height=400)

        # Titles and labels
        self.T = tk.Label(self.root, text="The Talking Hands", font=("Courier", 30, "bold"))
        self.T.place(x=60, y=5)

        self.T1 = tk.Label(self.root, text="Character :", font=("Courier", 30, "bold"))
        self.T1.place(x=10, y=580)
        self.panel3 = tk.Label(self.root, text=self.current_symbol, font=("Courier", 30))
        self.panel3.place(x=280, y=585)

        self.T3 = tk.Label(self.root, text="Sentence :", font=("Courier", 30, "bold"))
        self.T3.place(x=10, y=632)
        self.panel5 = tk.Label(self.root, text=self.str, font=("Courier", 30), wraplength=1025)
        self.panel5.place(x=260, y=632)

        # Translated sentence display
        self.Ttrans = tk.Label(self.root, text="Translated :", font=("Courier", 20, "bold"))
        self.Ttrans.place(x=700, y=530)
        self.trans_label = tk.Label(self.root, text="", font=("Courier", 18), wraplength=420, fg="blue")
        self.trans_label.place(x=700, y=560)

        # suggestion buttons
        self.b1 = tk.Button(self.root, text=self.word1, font=("Courier", 20), wraplength=825, command=self.action1)
        self.b1.place(x=390, y=700)
        self.b2 = tk.Button(self.root, text=self.word2, font=("Courier", 20), wraplength=825, command=self.action2)
        self.b2.place(x=590, y=700)
        self.b3 = tk.Button(self.root, text=self.word3, font=("Courier", 20), wraplength=825, command=self.action3)
        self.b3.place(x=790, y=700)
        self.b4 = tk.Button(self.root, text=self.word4, font=("Courier", 20), wraplength=825, command=self.action4)
        self.b4.place(x=990, y=700)

        # Speak & Clear buttons
        self.speak_btn = tk.Button(self.root, text="Speak", font=("Courier", 20), wraplength=100, command=self.speak_fun)
        self.speak_btn.place(x=1290, y=630)
        self.clear_btn = tk.Button(self.root, text="Clear", font=("Courier", 20), wraplength=100, command=self.clear_fun)
        self.clear_btn.place(x=1190, y=630)

        # Translate dropdown and label (bottom-right)
        self.language_label = tk.Label(self.root, text="Translate to:", font=("Courier", 18, "bold"))
        self.language_label.place(x=1180, y=560)

        self.language_var = tk.StringVar(value="English")
        self.language_dropdown = ttk.Combobox(self.root, textvariable=self.language_var, font=("Courier", 14), state="readonly", width=20)
        self.language_dropdown['values'] = list(self.lang_map.keys())
        self.language_dropdown.place(x=1180, y=590)
        self.language_dropdown.bind("<<ComboboxSelected>>", self.on_language_change)

        # initial translated label blank
        self.translated_sentence = ""
        self.trans_label.config(text="")

    # ------------------- VIDEO LOOP & PREDICTION -------------------
    def video_loop(self):
        try:
            ok, frame = self.vs.read()
            if not ok or frame is None:
                # Try again later
                self.root.after(50, self.video_loop)
                return

            # mirror image
            cv2image = cv2.flip(frame, 1)
            # copy for cropping
            cv2image_copy = np.array(cv2image)

            # convert BGR->RGB for display in tkinter
            display_img = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
            self.current_image = Image.fromarray(display_img)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            # find hands
            hands,_ = self.hd.findHands(cv2image, draw=False, flipType=True)  # returns list or []

            if hands:
                try:
                    hand = hands[0]
                    bbox = hand['bbox']
                    x, y, w, h = bbox
                    # safe crop coords
                    y1 = max(0, y - OFFSET)
                    y2 = min(cv2image_copy.shape[0], y + h + OFFSET)
                    x1 = max(0, x - OFFSET)
                    x2 = min(cv2image_copy.shape[1], x + w + OFFSET)
                    image = cv2image_copy[y1:y2, x1:x2]

                    if image is not None and image.size != 0:
                        handz, _ = self.hd2.findHands(image, draw=False, flipType=True)
                        if handz:
                            hand2 = handz[0]
                            handmap = hand2
                            self.pts = handmap['lmList']
                            # construct skeleton image on white base
                            white = safe_load_white().copy()
                            os_x = ((IMG_SIZE - w) // 2) - 15
                            os_y = ((IMG_SIZE - h) // 2) - 15

                            # draw lines - replicate your drawing logic
                            try:
                                for t in range(0, 4, 1):
                                    cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y),
                                             (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                                for t in range(5, 8, 1):
                                    cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y),
                                             (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                                for t in range(9, 12, 1):
                                    cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y),
                                             (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                                for t in range(13, 16, 1):
                                    cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y),
                                             (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                                for t in range(17, 20, 1):
                                    cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y),
                                             (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)

                                cv2.line(white, (self.pts[5][0] + os_x, self.pts[5][1] + os_y),
                                         (self.pts[9][0] + os_x, self.pts[9][1] + os_y), (0, 255, 0), 3)
                                cv2.line(white, (self.pts[9][0] + os_x, self.pts[9][1] + os_y),
                                         (self.pts[13][0] + os_x, self.pts[13][1] + os_y), (0, 255, 0), 3)
                                cv2.line(white, (self.pts[13][0] + os_x, self.pts[13][1] + os_y),
                                         (self.pts[17][0] + os_x, self.pts[17][1] + os_y), (0, 255, 0), 3)
                                cv2.line(white, (self.pts[0][0] + os_x, self.pts[0][1] + os_y),
                                         (self.pts[5][0] + os_x, self.pts[5][1] + os_y), (0, 255, 0), 3)
                                cv2.line(white, (self.pts[0][0] + os_x, self.pts[0][1] + os_y),
                                         (self.pts[17][0] + os_x, self.pts[17][1] + os_y), (0, 255, 0), 3)

                                for i in range(21):
                                    cv2.circle(white, (self.pts[i][0] + os_x, self.pts[i][1] + os_y), 2, (0, 0, 255), 1)
                            except Exception:
                                # if pts incomplete, skip drawing
                                pass

                            # update skeleton preview in UI
                            try:
                                disp = cv2.cvtColor(white, cv2.COLOR_BGR2RGB)
                                self.current_image2 = Image.fromarray(disp)
                                imgtk2 = ImageTk.PhotoImage(image=self.current_image2)
                                self.panel2.imgtk = imgtk2
                                self.panel2.config(image=imgtk2)
                            except Exception:
                                pass

                            # Predict letter & update UI
                            self.predict(white)

                except Exception as e:
                    # safe ignore detection errors; print trace for debug
                    # print("Detection error:", e)
                    # traceback.print_exc()
                    pass

            # update UI display strings
            self.panel3.config(text=str(self.current_symbol), font=("Courier", 30))
            self.panel5.config(text=self.str, font=("Courier", 30), wraplength=1025)
            # update suggestion buttons text with visual feedback - force update
            self.b1.config(text=self.word1)
            self.b2.config(text=self.word2)
            self.b3.config(text=self.word3)
            self.b4.config(text=self.word4)
            # Update background colors separately to ensure they take effect
            self.b1.config(bg="lightblue" if self.word1.strip() else "SystemButtonFace")
            self.b2.config(bg="lightblue" if self.word2.strip() else "SystemButtonFace")
            self.b3.config(bg="lightblue" if self.word3.strip() else "SystemButtonFace")
            self.b4.config(bg="lightblue" if self.word4.strip() else "SystemButtonFace")
            # Force UI update
            self.root.update_idletasks()

        except Exception:
            print("Video loop exception:", traceback.format_exc())

        finally:
            self.root.after(1, self.video_loop)

    # ------------------- UTILS: distance & actions -------------------
    def distance(self, x, y):
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    def action1(self):
        if self.word1.strip():
            idx_space = self.str.rfind(" ")
            # Get the current word we're trying to replace
            current_word = self.str[idx_space+1:] if idx_space >= 0 else self.str
            # Replace the current word with the suggestion
            if idx_space >= 0:
                self.str = self.str[:idx_space+1] + self.word1
            else:
                self.str = self.word1
            print(f"Replaced '{current_word}' with '{self.word1}'")
            self.update_translation_label()

    def action2(self):
        if self.word2.strip():
            idx_space = self.str.rfind(" ")
            # Get the current word we're trying to replace
            current_word = self.str[idx_space+1:] if idx_space >= 0 else self.str
            # Replace the current word with the suggestion
            if idx_space >= 0:
                self.str = self.str[:idx_space+1] + self.word2
            else:
                self.str = self.word2
            print(f"Replaced '{current_word}' with '{self.word2}'")
            self.update_translation_label()

    def action3(self):
        if self.word3.strip():
            idx_space = self.str.rfind(" ")
            # Get the current word we're trying to replace
            current_word = self.str[idx_space+1:] if idx_space >= 0 else self.str
            # Replace the current word with the suggestion
            if idx_space >= 0:
                self.str = self.str[:idx_space+1] + self.word3
            else:
                self.str = self.word3
            print(f"Replaced '{current_word}' with '{self.word3}'")
            self.update_translation_label()

    def action4(self):
        if self.word4.strip():
            idx_space = self.str.rfind(" ")
            # Get the current word we're trying to replace
            current_word = self.str[idx_space+1:] if idx_space >= 0 else self.str
            # Replace the current word with the suggestion
            if idx_space >= 0:
                self.str = self.str[:idx_space+1] + self.word4
            else:
                self.str = self.word4
            print(f"Replaced '{current_word}' with '{self.word4}'")
            self.update_translation_label()

    def speak_fun(self):
        """Translate current sentence (if not already) and speak it in selected language (gTTS)."""
        text_to_speak = self.str.strip()
        if len(text_to_speak) == 0:
            self.tts_engine.say("No text to speak")
            self.tts_engine.runAndWait()
            return

        # ensure translated_sentence exists for selected language; if not, translate
        try:
            selected_lang = self.language_var.get()
            lang_code = self.lang_map.get(selected_lang, "en")
            self.selected_lang_code = lang_code

            # Use existing translation if available, otherwise translate
            if self.translated_sentence and self.translated_sentence.strip():
                translated = self.translated_sentence
            else:
                # Translate synchronously (quick)
                translated = self.translate_text(text_to_speak, dest=lang_code)
                if translated is None or translated.startswith("[Translation unavailable]"):
                    # fallback to original
                    translated = text_to_speak

                # update UI translated label
                self.translated_sentence = translated
                self.trans_label.config(text=translated)

            # generate speech with gTTS in a separate thread to avoid blocking UI
            t = threading.Thread(target=self.play_tts, args=(translated, lang_code), daemon=True)
            t.start()

        except Exception as e:
            print("Speak error:", e)
            traceback.print_exc()
            # fallback to pyttsx3 in English
            try:
                self.tts_engine.say(text_to_speak)
                self.tts_engine.runAndWait()
            except Exception as e2:
                print("TTS engine error:", e2)

    def play_tts(self, text, lang_code):
        """Synthesize via gTTS then play audio. Uses playsound (blocking) inside its own thread."""
        if not text or not text.strip():
            print("No text to speak")
            return False
            
        print(f"Attempting to speak: '{text}' in language code: {lang_code}")
        
        # First try gTTS
        try:
            # Fix language code for gTTS compatibility
            use_code = lang_code
            if lang_code == 'zh-cn':
                use_code = 'zh-CN'
            elif lang_code == 'te':
                # Special handling for Telugu
                print("Using special handling for Telugu language")
                
            # Just try to create the gTTS object
            print(f"Using gTTS with language code: {use_code}")
            tts = gTTS(text=text, lang=use_code, slow=False)
            
            # Save to temporary file
            tmp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_audio.mp3")
            print(f"Saving audio to file: {tmp_path}")
            tts.save(tmp_path)
            
            # Play the audio
            print("Playing audio...")
            playsound.playsound(tmp_path)
            print("Audio playback completed successfully")
            
            # Try to clean up the temporary file
            try:
                os.remove(tmp_path)
                print(f"Removed temporary file: {tmp_path}")
            except Exception as e:
                print(f"Failed to remove temporary file: {e}")
                
            return True
            
        except Exception as e:
            print(f"gTTS error: {e}")
            print("Falling back to pyttsx3 for speech synthesis")
            
            # Fallback to pyttsx3
            try:
                print(f"Using pyttsx3 to speak: '{text}'")
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                print("pyttsx3 speech completed")
                return True
            except Exception as e2:
                print(f"pyttsx3 error: {e2}")
                return False
            
            # Clean up temporary file
            time.sleep(0.2)  # small delay to ensure file is not locked on Windows
            try:
                os.remove(tmp_path)
                print(f"Removed temporary file: {tmp_path}")
            except Exception as e:
                print(f"Failed to remove temporary file: {e}")
                
            return True  # Success, no need for fallback
            
        except Exception as e:
            print(f"gTTS/playback error: {e}")
            print("Falling back to pyttsx3...")
            
        # Fallback to pyttsx3
        try:
            print("Using pyttsx3 fallback")
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            print("pyttsx3 playback completed")
        except Exception as e:
            print(f"pyttsx3 fallback failed: {e}")
            print("All speech methods failed")
            pass

    def translate_text(self, text, dest='en'):
        """Translate with googletrans; returns translated string or None."""
        if not text or not text.strip():
            return None
            
        if self.translator is None:
            # If translator is not available, return original text
            return text
            
        try:
            res = self.translator.translate(text, dest=dest)
            if res and hasattr(res, 'text'):
                return res.text
            return text  # Fallback to original text if translation failed
        except Exception as e:
            print("Translation error:", e)
            return text  # Return original text on error

    def update_translation_label(self):
        """Translate current sentence into the chosen language and update UI label."""
        try:
            text_to_translate = self.str.strip()
            if not text_to_translate:
                self.translated_sentence = None
                self.trans_label.config(text="")
                return
                
            selected = self.language_var.get()
            lang_code = self.lang_map.get(selected, 'en')
            translated = self.translate_text(text_to_translate, dest=lang_code)
            
            # If translation returns the same as input for non-English target, it likely failed
            if translated == text_to_translate and lang_code != 'en':
                self.translated_sentence = f"[Translation unavailable] {text_to_translate}"
            else:
                self.translated_sentence = translated
                
            self.trans_label.config(text=self.translated_sentence or "")
        except Exception as e:
            print("Update translation label error:", e)
            # Don't clear the label on error if we already have content
            if not self.translated_sentence:
                self.trans_label.config(text="[Translation error]")


    def on_language_change(self, event):
        """When dropdown changes — update translated label immediately."""
        self.update_translation_label()

    def clear_fun(self):
        self.str = " "
        self.word1 = self.word2 = self.word3 = self.word4 = " "
        self.current_symbol = " "
        self.panel5.config(text=self.str)
        self.translated_sentence = None
        self.trans_label.config(text="")

    # ------------------- MODEL PREDICTION (your 8-group & subgroups logic) -------------------
    def predict(self, test_image):
        """This method keeps your original long logic intact but slightly adapted to class fields."""
        try:
            # Process every frame for better detection accuracy
            # Removed frame skipping to improve gesture detection
                
            white = test_image
            white = white.reshape(1, IMG_SIZE, IMG_SIZE, 3)
            prob = np.array(self.model.predict(white)[0], dtype='float32')

            ch1 = int(np.argmax(prob))
            prob[ch1] = 0
            ch2 = int(np.argmax(prob))
            prob[ch2] = 0
            ch3 = int(np.argmax(prob))
            prob[ch3] = 0

            pl = [ch1, ch2]

            # Now apply the full condition block (copied from your original final_pred.py)
            # ---------- Group corrections (many if-lists) ----------
            # condition for [Aemnst]
            l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
                 [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
                 [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
            if pl in l:
                if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                    ch1 = 0

            # condition for [o][s]
            l = [[2, 2], [2, 1]]
            if pl in l:
                if (self.pts[5][0] < self.pts[4][0]):
                    ch1 = 0

            # condition for [c0][aemnst]
            l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[4][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and self.pts[5][0] > self.pts[4][0]:
                    ch1 = 2

            # condition for [c0][aemnst]
            l = [[6, 0], [6, 6], [6, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if self.distance(self.pts[8], self.pts[16]) < 52:
                    ch1 = 2

            # condition for [gh][bdfikruvw]
            l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[6][1] > self.pts[8][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1] and self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                    ch1 = 3

            # con for [gh][l]
            l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[4][0] > self.pts[0][0]:
                    ch1 = 3

            # con for [gh][pqz]
            l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[2][1] + 15 < self.pts[16][1]:
                    ch1 = 3

            # con for [l][x]
            l = [[6, 4], [6, 1], [6, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if self.distance(self.pts[4], self.pts[11]) > 55:
                    ch1 = 4

            # con for [l][d]
            l = [[1, 4], [1, 6], [1, 1]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.distance(self.pts[4], self.pts[11]) > 50) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                    ch1 = 4

            # con for [l][gh]
            l = [[3, 6], [3, 4]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[4][0] < self.pts[0][0]):
                    ch1 = 4

            # con for [l][c0]
            l = [[2, 2], [2, 5], [2, 4]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[1][0] < self.pts[12][0]):
                    ch1 = 4

            # con for [gh][z]
            l = [[3, 6], [3, 5], [3, 4]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and self.pts[4][1] > self.pts[10][1]:
                    ch1 = 5

            # con for [gh][pq]
            l = [[3, 2], [3, 1], [3, 6]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[4][1] + 17 > self.pts[8][1] and self.pts[4][1] + 17 > self.pts[12][1] and self.pts[4][1] + 17 > self.pts[16][1] and self.pts[4][1] + 17 > self.pts[20][1]:
                    ch1 = 5

            # con for [l][pqz]
            l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[4][0] > self.pts[0][0]:
                    ch1 = 5

            # con for [pqz][aemnst]
            l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                    ch1 = 5

            # con for [pqz][yj]
            l = [[5, 7], [5, 2], [5, 6]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[3][0] < self.pts[0][0]:
                    ch1 = 7

            # con for [l][yj]
            l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[6][1] < self.pts[8][1]:
                    ch1 = 7

            # con for [x][yj]
            l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[18][1] > self.pts[20][1]:
                    ch1 = 7

            # condition for [x][aemnst]
            l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[5][0] > self.pts[16][0]:
                    ch1 = 6

            # condition for [yj][x]
            l = [[7, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[18][1] < self.pts[20][1] and self.pts[8][1] < self.pts[10][1]:
                    ch1 = 6

            # condition for [c0][x]
            l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
            pl = [ch1, ch2]
            if pl in l:
                if self.distance(self.pts[8], self.pts[16]) > 50:
                    ch1 = 6

            # con for [l][x]
            l = [[4, 6], [4, 2], [4, 1], [4, 4]]
            pl = [ch1, ch2]
            if pl in l:
                if self.distance(self.pts[4], self.pts[11]) < 60:
                    ch1 = 6

            # con for [x][d]
            l = [[1, 4], [1, 6], [1, 0], [1, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[5][0] - self.pts[4][0] - 15 > 0:
                    ch1 = 6

            # con for [b][pqz]
            l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0], [6, 3], [6, 4], [7, 5], [7, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                    ch1 = 1

            # con for [f][pqz]
            l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                    ch1 = 1

            l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                    ch1 = 1

            # con for [d][pqz]
            l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
            pl = [ch1, ch2]
            if pl in l:
                if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[4][1] > self.pts[14][1]):
                    ch1 = 1

            l = [[4, 1], [4, 2], [4, 4]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.distance(self.pts[4], self.pts[11]) < 50) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                    ch1 = 1

            l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
            pl = [ch1, ch2]
            if pl in l:
                if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[14][1] < self.pts[4][1]):
                    ch1 = 1

            l = [[6, 6], [6, 4], [6, 1], [6, 2]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[5][0] - self.pts[4][0] - 15 < 0:
                    ch1 = 1

            # con for [i][pqz]
            l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
            pl = [ch1, ch2]
            if pl in l:
                if ((self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1])):
                    ch1 = 1

            # con for [yj][bfdi]
            l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
            pl = [ch1, ch2]
            if pl in l:
                if (self.pts[4][0] < self.pts[5][0] + 15) and ((self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1])):
                    ch1 = 7

            # con for [uvr]
            l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
            pl = [ch1, ch2]
            if pl in l:
                if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1])) and self.pts[4][1] > self.pts[14][1]:
                    ch1 = 1

            # con for [w]
            fg = 13
            l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
            pl = [ch1, ch2]
            if pl in l:
                if not (self.pts[0][0] + fg < self.pts[8][0] and self.pts[0][0] + fg < self.pts[12][0] and self.pts[0][0] + fg < self.pts[16][0] and self.pts[0][0] + fg < self.pts[20][0]) and not (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and self.distance(self.pts[4], self.pts[11]) < 50:
                    ch1 = 1

            l = [[5, 0], [5, 5], [0, 1]]
            pl = [ch1, ch2]
            if pl in l:
                if self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1]:
                    ch1 = 1

            # -------------------------condn for 8 groups  ends

            # -------------------------condn for subgroups  starts
            if ch1 == 0:
                ch1 = 'S'
                if self.pts[4][0] < self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0]:
                    ch1 = 'A'
                if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]:
                    ch1 = 'T'
                if self.pts[4][1] > self.pts[8][1] and self.pts[4][1] > self.pts[12][1] and self.pts[4][1] > self.pts[16][1] and self.pts[4][1] > self.pts[20][1]:
                    ch1 = 'E'
                if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][0] > self.pts[14][0] and self.pts[4][1] < self.pts[18][1]:
                    ch1 = 'M'
                if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][1] < self.pts[18][1] and self.pts[4][1] < self.pts[14][1]:
                    ch1 = 'N'

            if ch1 == 2:
                if self.distance(self.pts[12], self.pts[4]) > 42:
                    ch1 = 'C'
                else:
                    ch1 = 'O'

            if ch1 == 3:
                if (self.distance(self.pts[8], self.pts[12])) > 72:
                    ch1 = 'G'
                else:
                    ch1 = 'H'

            if ch1 == 7:
                if self.distance(self.pts[8], self.pts[4]) > 42:
                    ch1 = 'Y'
                else:
                    ch1 = 'J'

            if ch1 == 4:
                ch1 = 'L'

            if ch1 == 6:
                ch1 = 'X'

            if ch1 == 5:
                if self.pts[4][0] > self.pts[12][0] and self.pts[4][0] > self.pts[16][0] and self.pts[4][0] > self.pts[20][0]:
                    if self.pts[8][1] < self.pts[5][1]:
                        ch1 = 'Z'
                    else:
                        ch1 = 'Q'
                else:
                    ch1 = 'P'

            if ch1 == 1:
                if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                    ch1 = 'B'
                if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                    ch1 = 'D'
                if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                    ch1 = 'F'
                if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                    ch1 = 'I'
                if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                    ch1 = 'W'
                if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and self.pts[4][1] < self.pts[9][1]:
                    ch1 = 'K'
                if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) < 8) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                    ch1 = 'U'
                if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) >= 8) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[4][1] > self.pts[9][1]):
                    ch1 = 'V'

                if (self.pts[8][0] > self.pts[12][0]) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                    ch1 = 'R'

            # special cases for space/next/backspace logic
            if ch1 in [1, 'E', 'S', 'X', 'Y', 'B']:
                if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                    ch1 = " "

            if ch1 in ['E', 'Y', 'B']:
                if (self.pts[4][0] < self.pts[5][0]) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                    ch1 = 'Next'

            # Improved backspace detection - more lenient conditions
            if ch1 in ['Next', 'B', 'C', 'H', 'F', 'X', 'Y', 'Z', 'L']:
                if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0]) and (self.pts[4][1] < self.pts[8][1] and self.pts[4][1] < self.pts[12][1]) and (self.pts[4][1] < self.pts[6][1] and self.pts[4][1] < self.pts[10][1]):
                    ch1 = 'Backspace'

            # finalize ch1 & update sentence state
            # The original code uses previous characters buffer and next/backspace logic
            # Keep behavior consistent:
            if isinstance(ch1, str):
                final_char = ch1
            else:
                final_char = str(ch1)

            # convert numbers that sneaked through to ascii mapping if needed
            # But original code already maps numeric groups to letters earlier.

            # Now push to buffer logic similar to original
            # Handle Next sign - add previous character to sentence
            if (final_char.lower() == "next" or final_char == "Next") and (self.prev_char.lower() != "next" and self.prev_char != "Next"):
                # Debug print to see what's happening
                print(f"Next detected! Previous chars: {self.ten_prev_char}")
                
                # Get the previous character (not 'Next')
                prev_idx = (self.count-2) % 10
                prev_char = self.ten_prev_char[prev_idx]
                
                # Check if the previous character is not 'Next' or 'Backspace'
                if prev_char != "Next" and prev_char != "Backspace" and prev_char.strip():
                    print(f"Adding character: {prev_char}")
                    self.str = self.str + prev_char
                    
                # Update translation after adding character
                self.update_translation_label()
                
            # Handle Backspace - remove last character from sentence
            if final_char == "Backspace" and self.prev_char != "Backspace":
                print("Backspace detected!")
                if len(self.str) > 0:
                    self.str = self.str[:-1]
                    print(f"String after backspace: '{self.str}'")
                    self.update_translation_label()

            if final_char == " " and self.prev_char != " ":
                self.str = self.str + " "

            self.prev_char = final_char
            self.current_symbol = final_char
            self.count += 1
            self.ten_prev_char[self.count % 10] = final_char

            # suggestion logic (word-level)
            if len(self.str.strip()) != 0:
                st = self.str.rfind(" ")
                ed = len(self.str)
                word = self.str[st+1:ed]
                self.word = word
                
                # Reset suggestion buttons to empty
                self.word1 = self.word2 = self.word3 = self.word4 = " "
                
                # Only try to get suggestions if we have a word and dictionary
                if len(word.strip()) != 0 and self.dict_en:
                    try:
                        print(f"Getting suggestions for word: '{word}'")
                        # Force lowercase for better suggestion matching
                        suggestions = self.dict_en.suggest(word.lower())
                        print(f"Got suggestions: {suggestions}")
                        
                        # Make sure we have suggestions before assigning
                        if suggestions and len(suggestions) > 0:
                            if len(suggestions) >= 1:
                                self.word1 = suggestions[0]
                            if len(suggestions) >= 2:
                                self.word2 = suggestions[1]
                            if len(suggestions) >= 3:
                                self.word3 = suggestions[2]
                            if len(suggestions) >= 4:
                                self.word4 = suggestions[3]
                                
                            # Debug print the suggestions
                            print(f"Setting suggestion buttons: '{self.word1}', '{self.word2}', '{self.word3}', '{self.word4}'")
                        else:
                            print("No suggestions found")
                    except Exception as e:
                        print(f"Error getting suggestions: {e}")
                        self.word1 = self.word2 = self.word3 = self.word4 = " "

            # update translation label if language selected
            self.update_translation_label()

        except Exception as e:
            print("Predict exception:", e)
            traceback.print_exc()

    # ------------------- CLEANUP -------------------
    def destructor(self):
        try:
            print("Cleaning up...")
            self.root.destroy()
            try:
                self.vs.release()
            except:
                pass
            cv2.destroyAllWindows()
        except:
            pass

# ------------------- RUN -------------------
if __name__ == "__main__":
    print("Starting Talking Hands (multilingual)...")
    app = Application()
    app.root.mainloop()