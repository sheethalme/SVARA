"""
prompts.py -- everything the voice call SAYS, in English, Hindi, Malayalam and Kannada
=====================================================================================

* ``say(key, lang, **values)``        a system sentence ("Is that correct?", "Thank you ...")
* ``default_question(field, lang)``   the default question for a built-in field
* ``field_label(field, lang)``        the spoken name of a field ("age", "उम्र", "പ്രായം", "ವಯಸ್ಸು")
* ``spoken_value(kind, value, lang)`` a value written for text-to-speech (digits one by one, month names ...)

HONEST NOTE: the Hindi, Malayalam and Kannada sentences are DRAFTS written without a native-speaker review.
Have a native speaker read them (they are spoken aloud to patients!).  Any question can be overridden per form
(in the form's JSON, field "question") and any sentence can be overridden with ``OVERRIDES`` or by editing this file.
Hindi sentences are worded to avoid gendered verb forms where possible.
"""

from __future__ import annotations

from typing import Dict, Optional

import extractor

LANGS = ("en", "hi", "ml", "kn")
DEFAULT_LANG = "en"

# ---------------------------------------------------------------------------------------- sentences
SENTENCES: Dict[str, Dict[str, str]] = {
    "greeting_consent": {
        "en": "Hello. I will ask you a few questions to fill in the {title} form. Your answers will be saved. "
              "Do you agree? Please say yes or no.",
        "hi": "नमस्ते। आपसे {title} फ़ॉर्म भरने के लिए कुछ सवाल पूछे जाएँगे। आपके जवाब सहेजे जाएँगे। "
              "क्या आप सहमत हैं? कृपया हाँ या नहीं बोलिए।",
        "ml": "നമസ്കാരം. {title} ഫോം പൂരിപ്പിക്കാൻ ഞാൻ ചില ചോദ്യങ്ങൾ ചോദിക്കാം. നിങ്ങളുടെ ഉത്തരങ്ങൾ സേവ് ചെയ്യും. "
              "നിങ്ങൾ സമ്മതിക്കുന്നുണ്ടോ? ദയവായി അതെ അല്ലെങ്കിൽ അല്ല എന്ന് പറയൂ.",
        "kn": "ನಮಸ್ಕಾರ. {title} ಫಾರ್ಮ್ ತುಂಬಲು ನಾನು ಕೆಲವು ಪ್ರಶ್ನೆಗಳನ್ನು ಕೇಳುತ್ತೇನೆ. ನಿಮ್ಮ ಉತ್ತರಗಳನ್ನು ಉಳಿಸಲಾಗುವುದು. "
              "ನೀವು ಒಪ್ಪುತ್ತೀರಾ? ದಯವಿಟ್ಟು ಹೌದು ಅಥವಾ ಇಲ್ಲ ಎಂದು ಹೇಳಿ.",
    },
    "greeting": {
        "en": "Hello. I will ask you a few questions to fill in the {title} form.",
        "hi": "नमस्ते। आपसे {title} फ़ॉर्म भरने के लिए कुछ सवाल पूछे जाएँगे।",
        "ml": "നമസ്കാരം. {title} ഫോം പൂരിപ്പിക്കാൻ ഞാൻ ചില ചോദ്യങ്ങൾ ചോദിക്കാം.",
        "kn": "ನಮಸ್ಕಾರ. {title} ಫಾರ್ಮ್ ತುಂಬಲು ನಾನು ಕೆಲವು ಪ್ರಶ್ನೆಗಳನ್ನು ಕೇಳುತ್ತೇನೆ.",
    },
    "consent_declined": {
        "en": "Okay, no problem. Goodbye.",
        "hi": "ठीक है, कोई बात नहीं। धन्यवाद।",
        "ml": "ശരി, കുഴപ്പമില്ല. നന്ദി.",
        "kn": "ಸರಿ, ಪರವಾಗಿಲ್ಲ. ಧನ್ಯವಾದಗಳು.",
    },
    "ask_yes_no": {
        "en": "Please say yes or no.",
        "hi": "कृपया हाँ या नहीं बोलिए।",
        "ml": "ദയവായി അതെ അല്ലെങ്കിൽ അല്ല എന്ന് പറയൂ.",
        "kn": "ದಯವಿಟ್ಟು ಹೌದು ಅಥವಾ ಇಲ್ಲ ಎಂದು ಹೇಳಿ.",
    },
    "didnt_hear": {
        "en": "I could not hear you.",
        "hi": "आपकी आवाज़ सुनाई नहीं दी।",
        "ml": "നിങ്ങളുടെ ശബ്ദം കേട്ടില്ല.",
        "kn": "ನಿಮ್ಮ ಧ್ವನಿ ಕೇಳಿಸಲಿಲ್ಲ.",
    },
    "didnt_catch": {
        "en": "Sorry, I did not catch that.",
        "hi": "माफ़ कीजिए, यह समझ नहीं आया।",
        "ml": "ക്ഷമിക്കണം, മനസ്സിലായില്ല.",
        "kn": "ಕ್ಷಮಿಸಿ, ಅರ್ಥವಾಗಲಿಲ್ಲ.",
    },
    "confirm": {
        "en": "I heard your {label} as {value}. Is that correct?",
        "hi": "मैंने {label} {value} सुना। क्या यह सही है?",
        "ml": "നിങ്ങളുടെ {label} {value} എന്നാണ് ഞാൻ കേട്ടത്. ഇത് ശരിയാണോ?",
        "kn": "ನಿಮ್ಮ {label} {value} ಎಂದು ನಾನು ಕೇಳಿದೆ. ಇದು ಸರಿಯೇ?",
    },
    "confirm_again": {
        "en": "Okay. So your {label} is {value}. Is that correct?",
        "hi": "ठीक है। तो {label} {value} है। क्या यह सही है?",
        "ml": "ശരി. അപ്പോൾ നിങ്ങളുടെ {label} {value} ആണ്. ഇത് ശരിയാണോ?",
        "kn": "ಸರಿ. ಹಾಗಾದರೆ ನಿಮ್ಮ {label} {value}. ಇದು ಸರಿಯೇ?",
    },
    "reask": {
        "en": "Okay. Please tell me your {label} again.",
        "hi": "ठीक है। कृपया {label} फिर से बताइए।",
        "ml": "ശരി. ദയവായി നിങ്ങളുടെ {label} ഒന്നുകൂടി പറയൂ.",
        "kn": "ಸರಿ. ದಯವಿಟ್ಟು ನಿಮ್ಮ {label} ಅನ್ನು ಮತ್ತೊಮ್ಮೆ ಹೇಳಿ.",
    },
    "thanks": {
        "en": "Thank you.",
        "hi": "धन्यवाद।",
        "ml": "നന്ദി.",
        "kn": "ಧನ್ಯವಾದಗಳು.",
    },
    "skipped": {
        "en": "Okay, let us skip this one.",
        "hi": "ठीक है, इसे छोड़ देते हैं।",
        "ml": "ശരി, ഇത് ഒഴിവാക്കാം.",
        "kn": "ಸರಿ, ಇದನ್ನು ಬಿಟ್ಟುಬಿಡೋಣ.",
    },
    "required": {
        "en": "I am sorry, this question is needed for the form.",
        "hi": "माफ़ कीजिए, फ़ॉर्म के लिए यह जानकारी ज़रूरी है।",
        "ml": "ക്ഷമിക്കണം, ഫോമിന് ഇത് ആവശ്യമാണ്.",
        "kn": "ಕ್ಷಮಿಸಿ, ಫಾರ್ಮ್‌ಗೆ ಇದು ಅಗತ್ಯವಿದೆ.",
    },
    "review_intro": {
        "en": "Here is the information I have.",
        "hi": "आपकी जानकारी यह है।",
        "ml": "നിങ്ങളുടെ വിവരങ്ങൾ ഇതാ.",
        "kn": "ನಿಮ್ಮ ಮಾಹಿತಿ ಇದು.",
    },
    "review_item": {
        "en": "{label}: {value}.",
        "hi": "{label}: {value}।",
        "ml": "{label}: {value}.",
        "kn": "{label}: {value}.",
    },
    "review_skipped": {
        "en": "{label}: skipped.",
        "hi": "{label}: छोड़ा गया।",
        "ml": "{label}: ഒഴിവാക്കി.",
        "kn": "{label}: ಬಿಟ್ಟುಬಿಡಲಾಗಿದೆ.",
    },
    "review_question": {
        "en": "Is everything correct?",
        "hi": "क्या सब कुछ सही है?",
        "ml": "എല്ലാം ശരിയാണോ?",
        "kn": "ಎಲ್ಲವೂ ಸರಿಯೇ?",
    },
    "which_change": {
        "en": "Which one would you like to change? Please say its name.",
        "hi": "आप किसे बदलना चाहेंगे? कृपया उसका नाम बोलिए।",
        "ml": "ഏതാണ് മാറ്റേണ്ടത്? ദയവായി അതിന്റെ പേര് പറയൂ.",
        "kn": "ಯಾವುದನ್ನು ಬದಲಾಯಿಸಬೇಕು? ದಯವಿಟ್ಟು ಅದರ ಹೆಸರನ್ನು ಹೇಳಿ.",
    },
    "which_change_again": {
        "en": "I could not tell which one. Please say its name, for example {example}.",
        "hi": "समझ नहीं आया कि कौन सा। कृपया नाम बोलिए, जैसे {example}।",
        "ml": "ഏതാണെന്ന് മനസ്സിലായില്ല. ദയവായി പേര് പറയൂ, ഉദാഹരണത്തിന് {example}.",
        "kn": "ಯಾವುದೆಂದು ಅರ್ಥವಾಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಹೆಸರನ್ನು ಹೇಳಿ, ಉದಾಹರಣೆಗೆ {example}.",
    },
    "done": {
        "en": "Thank you. Your form has been saved. Goodbye.",
        "hi": "धन्यवाद। आपका फ़ॉर्म सहेज लिया गया है। नमस्ते।",
        "ml": "നന്ദി. നിങ്ങളുടെ ഫോം സേവ് ചെയ്തു. നമസ്കാരം.",
        "kn": "ಧನ್ಯವಾದಗಳು. ನಿಮ್ಮ ಫಾರ್ಮ್ ಉಳಿಸಲಾಗಿದೆ. ನಮಸ್ಕಾರ.",
    },
    "timeout": {
        "en": "I could not hear you for a while, so I am ending the call. Your answers so far are saved. "
              "Please call again.",
        "hi": "काफ़ी देर से आवाज़ नहीं आई, इसलिए कॉल समाप्त की जा रही है। अब तक के जवाब सहेजे गए हैं। कृपया फिर से कॉल कीजिए।",
        "ml": "കുറച്ചുനേരമായി ശബ്ദം കേട്ടില്ല, അതിനാൽ കോൾ അവസാനിപ്പിക്കുന്നു. ഇതുവരെയുള്ള ഉത്തരങ്ങൾ സേവ് ചെയ്തിട്ടുണ്ട്. "
              "ദയവായി വീണ്ടും വിളിക്കൂ.",
        "kn": "ಸ್ವಲ್ಪ ಸಮಯದಿಂದ ಧ್ವನಿ ಕೇಳಿಸಿಲ್ಲ, ಆದ್ದರಿಂದ ಕರೆಯನ್ನು ಮುಗಿಸುತ್ತಿದ್ದೇನೆ. ಇಲ್ಲಿಯವರೆಗಿನ ಉತ್ತರಗಳನ್ನು ಉಳಿಸಲಾಗಿದೆ. "
              "ದಯವಿಟ್ಟು ಮತ್ತೆ ಕರೆ ಮಾಡಿ.",
    },
    "trouble": {
        "en": "I am having trouble understanding you. A person will help you. Your answers so far are saved.",
        "hi": "आपकी बात समझने में दिक्कत हो रही है। कोई व्यक्ति आपकी मदद करेगा। अब तक के जवाब सहेजे गए हैं।",
        "ml": "നിങ്ങൾ പറയുന്നത് മനസ്സിലാക്കാൻ ബുദ്ധിമുട്ടുണ്ട്. ഒരാൾ നിങ്ങളെ സഹായിക്കും. ഇതുവരെയുള്ള ഉത്തരങ്ങൾ സേവ് ചെയ്തിട്ടുണ്ട്.",
        "kn": "ನಿಮ್ಮ ಮಾತು ಅರ್ಥಮಾಡಿಕೊಳ್ಳಲು ತೊಂದರೆಯಾಗುತ್ತಿದೆ. ಒಬ್ಬರು ನಿಮಗೆ ಸಹಾಯ ಮಾಡುತ್ತಾರೆ. ಇಲ್ಲಿಯವರೆಗಿನ ಉತ್ತರಗಳನ್ನು ಉಳಿಸಲಾಗಿದೆ.",
    },
    "error": {
        "en": "Sorry, something went wrong. Let me ask again.",
        "hi": "माफ़ कीजिए, कुछ गड़बड़ हो गई। फिर से पूछा जा रहा है।",
        "ml": "ക്ഷമിക്കണം, എന്തോ തകരാർ സംഭവിച്ചു. വീണ്ടും ചോദിക്കാം.",
        "kn": "ಕ್ಷಮಿಸಿ, ಏನೋ ತೊಂದರೆಯಾಯಿತು. ಮತ್ತೆ ಕೇಳುತ್ತೇನೆ.",
    },
    "hangup": {
        "en": "The call has ended. Your answers so far are saved.",
        "hi": "कॉल समाप्त हो गई है। अब तक के जवाब सहेजे गए हैं।",
        "ml": "കോൾ അവസാനിച്ചു. ഇതുവരെയുള്ള ഉത്തരങ്ങൾ സേവ് ചെയ്തിട്ടുണ്ട്.",
        "kn": "ಕರೆ ಮುಗಿದಿದೆ. ಇಲ್ಲಿಯವರೆಗಿನ ಉತ್ತರಗಳನ್ನು ಉಳಿಸಲಾಗಿದೆ.",
    },
    "generic_question": {
        "en": "Please tell me your {label}.",
        "hi": "कृपया अपना {label} बताइए।",
        "ml": "ദയവായി നിങ്ങളുടെ {label} പറയൂ.",
        "kn": "ದಯವಿಟ್ಟು ನಿಮ್ಮ {label} ಅನ್ನು ಹೇಳಿ.",
    },
}

# a short hint given to the recogniser on yes/no turns (Whisper "prompt": makes short answers more likely to be read as such)
CONFIRM_HINT = {"en": "Yes. No. Correct. Wrong. Say again.", "hi": "हाँ। नहीं। सही। गलत। फिर से बोलिए।",
                "ml": "അതെ. അല്ല. ശരി. തെറ്റ്. ഒന്നുകൂടി പറയൂ.", "kn": "ಹೌದು. ಇಲ್ಲ. ಸರಿ. ತಪ್ಪು. ಮತ್ತೆ ಹೇಳಿ."}

OVERRIDES: Dict[str, Dict[str, str]] = {}          # {"done": {"en": "..."}}  (set from your own code)

# ---------------------------------------------------------------------------------------- built-in fields
LABELS: Dict[str, Dict[str, str]] = {
    "name": {"en": "name", "hi": "नाम", "ml": "പേര്", "kn": "ಹೆಸರು"},
    "age": {"en": "age", "hi": "उम्र", "ml": "പ്രായം", "kn": "ವಯಸ್ಸು"},
    "gender": {"en": "gender", "hi": "लिंग", "ml": "ലിംഗം", "kn": "ಲಿಂಗ"},
    "phone": {"en": "phone number", "hi": "फ़ोन नंबर", "ml": "ഫോൺ നമ്പർ", "kn": "ಫೋನ್ ಸಂಖ್ಯೆ"},
    "date_of_birth": {"en": "date of birth", "hi": "जन्म तिथि", "ml": "ജനനത്തീയതി", "kn": "ಹುಟ್ಟಿದ ದಿನಾಂಕ"},
    "date": {"en": "date", "hi": "तारीख", "ml": "തീയതി", "kn": "ದಿನಾಂಕ"},
    "address": {"en": "address", "hi": "पता", "ml": "വിലാസം", "kn": "ವಿಳಾಸ"},
    "pincode": {"en": "PIN code", "hi": "पिन कोड", "ml": "പിൻ കോഡ്", "kn": "ಪಿನ್ ಕೋಡ್"},
    "aadhaar": {"en": "Aadhaar number", "hi": "आधार नंबर", "ml": "ആധാർ നമ്പർ", "kn": "ಆಧಾರ್ ಸಂಖ್ಯೆ"},
    "blood_group": {"en": "blood group", "hi": "ब्लड ग्रुप", "ml": "ബ്ലഡ് ഗ്രൂപ്പ്", "kn": "ರಕ್ತದ ಗುಂಪು"},
    "email": {"en": "email address", "hi": "ईमेल पता", "ml": "ഇമെയിൽ വിലാസം", "kn": "ಇಮೇಲ್ ವಿಳಾಸ"},
    "complaint": {"en": "problem", "hi": "समस्या", "ml": "പ്രശ്നം", "kn": "ಸಮಸ್ಯೆ"},
    "occupation": {"en": "occupation", "hi": "काम", "ml": "ജോലി", "kn": "ಉದ್ಯೋಗ"},
    "yes_no": {"en": "answer", "hi": "जवाब", "ml": "ഉത്തരം", "kn": "ಉತ್ತರ"},
    "amount": {"en": "amount", "hi": "राशि", "ml": "തുക", "kn": "ಮೊತ್ತ"},
}

QUESTIONS: Dict[str, Dict[str, str]] = {
    "name": {"en": "What is your full name?", "hi": "आपका पूरा नाम क्या है?",
             "ml": "നിങ്ങളുടെ പൂർണ്ണമായ പേര് എന്താണ്?", "kn": "ನಿಮ್ಮ ಪೂರ್ಣ ಹೆಸರು ಏನು?"},
    "age": {"en": "How old are you?", "hi": "आपकी उम्र कितनी है?",
            "ml": "നിങ്ങൾക്ക് എത്ര വയസ്സായി?", "kn": "ನಿಮ್ಮ ವಯಸ್ಸು ಎಷ್ಟು?"},
    "gender": {"en": "What is your gender? Male, female, or other?", "hi": "आपका लिंग क्या है? पुरुष, महिला या अन्य?",
               "ml": "നിങ്ങളുടെ ലിംഗം എന്താണ്? പുരുഷൻ, സ്ത്രീ, അല്ലെങ്കിൽ മറ്റുള്ളവ?",
               "kn": "ನಿಮ್ಮ ಲಿಂಗ ಯಾವುದು? ಪುರುಷ, ಮಹಿಳೆ ಅಥವಾ ಇತರೆ?"},
    "phone": {"en": "What is your mobile number? Please say the digits one by one.",
              "hi": "आपका मोबाइल नंबर क्या है? कृपया अंक एक-एक करके बोलिए।",
              "ml": "നിങ്ങളുടെ മൊബൈൽ നമ്പർ എന്താണ്? ദയവായി അക്കങ്ങൾ ഓരോന്നായി പറയൂ.",
              "kn": "ನಿಮ್ಮ ಮೊಬೈಲ್ ಸಂಖ್ಯೆ ಏನು? ದಯವಿಟ್ಟು ಅಂಕೆಗಳನ್ನು ಒಂದೊಂದಾಗಿ ಹೇಳಿ."},
    "date_of_birth": {"en": "What is your date of birth? For example, five March nineteen ninety.",
                      "hi": "आपकी जन्म तिथि क्या है? उदाहरण के लिए, पाँच मार्च उन्नीस सौ नब्बे।",
                      "ml": "നിങ്ങളുടെ ജനനത്തീയതി എന്താണ്? ഉദാഹരണത്തിന്, 1990 മാർച്ച് 5.",
                      "kn": "ನಿಮ್ಮ ಹುಟ್ಟಿದ ದಿನಾಂಕ ಯಾವುದು? ಉದಾಹರಣೆಗೆ, 1990 ಮಾರ್ಚ್ 5."},
    "date": {"en": "Which date?", "hi": "कौन सी तारीख?", "ml": "ഏത് തീയതി?", "kn": "ಯಾವ ದಿನಾಂಕ?"},
    "address": {"en": "What is your address?", "hi": "आपका पता क्या है?",
                "ml": "നിങ്ങളുടെ വിലാസം എന്താണ്?", "kn": "ನಿಮ್ಮ ವಿಳಾಸ ಯಾವುದು?"},
    "pincode": {"en": "What is your PIN code?", "hi": "आपका पिन कोड क्या है?",
                "ml": "നിങ്ങളുടെ പിൻ കോഡ് എന്താണ്?", "kn": "ನಿಮ್ಮ ಪಿನ್ ಕೋಡ್ ಯಾವುದು?"},
    "aadhaar": {"en": "What is your Aadhaar number? Please say the twelve digits one by one.",
                "hi": "आपका आधार नंबर क्या है? कृपया बारह अंक एक-एक करके बोलिए।",
                "ml": "നിങ്ങളുടെ ആധാർ നമ്പർ എന്താണ്? ദയവായി പന്ത്രണ്ട് അക്കങ്ങൾ ഓരോന്നായി പറയൂ.",
                "kn": "ನಿಮ್ಮ ಆಧಾರ್ ಸಂಖ್ಯೆ ಏನು? ದಯವಿಟ್ಟು ಹನ್ನೆರಡು ಅಂಕೆಗಳನ್ನು ಒಂದೊಂದಾಗಿ ಹೇಳಿ."},
    "blood_group": {"en": "What is your blood group? For example, A positive.",
                    "hi": "आपका ब्लड ग्रुप क्या है? उदाहरण के लिए, ए पॉजिटिव।",
                    "ml": "നിങ്ങളുടെ ബ്ലഡ് ഗ്രൂപ്പ് ഏതാണ്? ഉദാഹരണത്തിന്, എ പോസിറ്റീവ്.",
                    "kn": "ನಿಮ್ಮ ರಕ್ತದ ಗುಂಪು ಯಾವುದು? ಉದಾಹರಣೆಗೆ, ಎ ಪಾಸಿಟಿವ್."},
    "email": {"en": "What is your email address?", "hi": "आपका ईमेल पता क्या है?",
              "ml": "നിങ്ങളുടെ ഇമെയിൽ വിലാസം എന്താണ്?", "kn": "ನಿಮ್ಮ ಇಮೇಲ್ ವಿಳಾಸ ಯಾವುದು?"},
    "complaint": {"en": "What is the problem you are visiting for?", "hi": "आप किस समस्या के लिए आए हैं?",
                  "ml": "നിങ്ങൾ എന്ത് പ്രശ്നത്തിനാണ് വന്നത്?", "kn": "ನೀವು ಯಾವ ಸಮಸ್ಯೆಗಾಗಿ ಬಂದಿದ್ದೀರಿ?"},
    "occupation": {"en": "What is your occupation?", "hi": "आप क्या काम करते हैं?",
                   "ml": "നിങ്ങളുടെ ജോലി എന്താണ്?", "kn": "ನಿಮ್ಮ ಉದ್ಯೋಗ ಯಾವುದು?"},
    "yes_no": {"en": "Please say yes or no.", "hi": "कृपया हाँ या नहीं बोलिए।",
               "ml": "ദയവായി അതെ അല്ലെങ്കിൽ അല്ല എന്ന് പറയൂ.", "kn": "ದಯವಿಟ್ಟು ಹೌದು ಅಥವಾ ಇಲ್ಲ ಎಂದು ಹೇಳಿ."},
    "amount": {"en": "What is the amount?", "hi": "राशि कितनी है?", "ml": "തുക എത്രയാണ്?", "kn": "ಮೊತ್ತ ಎಷ್ಟು?"},
}

_GENDER_WORDS = {"Male": {"en": "male", "hi": "पुरुष", "ml": "പുരുഷൻ", "kn": "ಪುರುಷ"},
                 "Female": {"en": "female", "hi": "महिला", "ml": "സ്ത്രീ", "kn": "ಮಹಿಳೆ"},
                 "Other": {"en": "other", "hi": "अन्य", "ml": "മറ്റുള്ളവ", "kn": "ಇತರೆ"}}
_YES_NO_WORDS = {"Yes": {"en": "yes", "hi": "हाँ", "ml": "അതെ", "kn": "ಹೌದು"},
                 "No": {"en": "no", "hi": "नहीं", "ml": "അല്ല", "kn": "ಇಲ್ಲ"}}

# words that mean "skip this question" (only honoured for optional fields)
SKIP_PHRASES = ("skip", "pass", "i don't know", "i do not know", "don't know", "not applicable", "no idea",
                "छोड़ दो", "छोड़ दीजिए", "पता नहीं", "छोड़िए", "വേണ്ട", "അറിയില്ല", "ഒഴിവാക്കൂ", "ಬೇಡ", "ಗೊತ್ತಿಲ್ಲ", "ಬಿಟ್ಟುಬಿಡಿ")


def norm_lang(lang: Optional[str]) -> str:
    l = (lang or DEFAULT_LANG).strip().lower().split("-")[0]
    return l if l in LANGS else DEFAULT_LANG


def say(key: str, lang: Optional[str], **values: object) -> str:
    """A system sentence in ``lang`` (falls back to English); ``{placeholders}`` are filled from ``values``."""
    lang = norm_lang(lang)
    table = {**SENTENCES[key], **OVERRIDES.get(key, {})}
    return table.get(lang, table["en"]).format(**values)


def field_label(field: str, lang: Optional[str], custom: Optional[Dict[str, str]] = None) -> str:
    lang = norm_lang(lang)
    if custom:
        return custom.get(lang) or custom.get("en") or next(iter(custom.values()))
    base = LABELS.get(field)
    if base:
        return base.get(lang, base["en"])
    return field.replace("_", " ")


def default_question(field: str, lang: Optional[str], label: Optional[str] = None) -> str:
    lang = norm_lang(lang)
    q = QUESTIONS.get(field)
    if q:
        return q.get(lang, q["en"])
    return say("generic_question", lang, label=label or field.replace("_", " "))


def spoken_value(kind: str, value: Optional[str], lang: Optional[str]) -> str:
    """How to read ``value`` aloud: digits one by one, dates with month names, gender / yes-no in the user's language."""
    if value is None:
        return ""
    lang = norm_lang(lang)
    if kind == "gender" and value in _GENDER_WORDS:
        return _GENDER_WORDS[value][lang]
    if kind == "yes_no" and value in _YES_NO_WORDS:
        return _YES_NO_WORDS[value][lang]
    if kind == "age":
        return value
    if kind in ("phone", "pincode", "aadhaar", "date", "date_of_birth", "blood_group", "email"):
        return extractor.spoken_form(kind if kind != "date_of_birth" else "date_of_birth", value, lang)
    return value
