# SVARA frontend outcomes

## 1. Responsive SVARA overview and navigation
- Create a responsive SVARA landing/dashboard experience that explains hands-free voice form filling and guides users toward starting a call.
- Use a visual design system using burgundy, pistachio green, warm sand, and bright red with soft contrast, subtle depth, responsive motion, pointer-reactive cards, and layered 3D/animated audio motifs.
- Use a refreshed SVARA signal mark with a burgundy tile, pistachio signal stroke, warm-sand highlight, and bright-red node across the favicon and application chrome.
- Support desktop, tablet, and mobile layouts, including keyboard focus states, reduced-motion support, and readable typography.

## 2. Form selection and pre-call setup
- Show available backend forms with descriptive metadata, field counts, and supported languages.
- Let the user choose a form and language before starting a call.
- Include clear microphone permission and accessibility guidance.

## 3. Hands-free voice call experience
- Use the existing start-call, turn, event, and text-to-speech APIs without requiring buttons during the active conversation.
- Show live call states for speaking, listening, hearing, thinking, errors, and completion, including microphone level and progress indicators.
- Provide an accessible conversation timeline showing SVARA prompts and recognized user responses.
- Support call completion, hang-up, retry, and recovery states that communicate whether the form was saved, interrupted, or needs attention.
- Include a graceful demo mode when the Python call server is not connected, without obscuring the real API integration.

## 4. Response tracking
- Use the existing calls API for call summaries, outcomes, timestamps, and links to inspect individual records.
- Provide a readable response detail view for selected calls.

## 5. Separate Admin and Caller experiences
- Provide a clear workspace switch between an Admin console and a Caller experience without duplicating the backend call flow.
- Admins can open a form studio, create a new form, edit an existing form, reorder fields, mark fields required or optional, add custom field kinds and spoken questions, and save drafts or publish changes.
- Callers see a focused home and form-selection experience with language choice, microphone guidance, hands-free calls, live progress, transcript, completion, and recovery states.

## 6. Delivery and verification
- Keep the existing backend contracts intact and make the frontend usable when served by the backend token-injection flow.
- Maintain `/manus-routes.json` for the current page routes.
- Verify the production build and runtime readiness before delivery.
