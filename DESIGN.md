# Interface design

Aftermeal uses a minimal interface with a centered heading, paired photographs and one result.

## Typography and color

The interface uses the native system sans-serif stack. Body copy is 17px, introductory copy 21px, controls 16px and secondary labels 14px. Page headings scale from 40px to 64px. There are no external font requests.

- Background: `#ffffff`; supporting surface: `#f5f5f7`.
- Primary text: `#1d1d1f`; secondary text: `#626267`.
- Action and selection: `#0066cc`; hover: `#0055aa`.
- Dividers: `#d2d2d7`; errors: `#b3261e`.

## Main workflow

Sample pairs are read-only. Choosing a sample calculates its estimate automatically, without an extra action button. Your photos exposes two upload targets and one Analyze pair button. Each empty upload target has one icon and label; a populated target has one text-only Change photo control. Upload targets accept file selection or drag-and-drop.

Sample photos show a switchable blue food mask, enabled by default and explicitly experimental. Cached overlays are separate from the percentage model; uploads generate masks locally through a separate endpoint and can toggle back to their originals. The result follows the photographs. Whole-percent estimates, source weight ratios and prediction errors remain visible; explanation and model details are inside About this estimate. Dish names label the three gallery samples; the difficult fourth pair expands inline within Accuracy. An empty upload has no result panel. Status text appears only when it provides useful loading, recovery or completion feedback.

Analyze, Accuracy and About are hash-addressable tabs with arrow-key, Home and End navigation. Keyboard focus remains visible. Reduced-motion preferences disable transitions.

## Weighed collection

Collect data is a separate persistent workflow at `/capture`. The empty state explains scale setup and starts a pilot session. A serving has one starting photo and tare, followed by one or more after readings. The scale-derived fraction is visibly separate from any model estimate. Corrections preserve prior readings; excluding a bad starting record preserves its history. Pilot progress counts completed independent servings rather than stages. Photos and readings are exported together. The collection's persistence is stated before saving.

## Responsive layout

Photos stay paired at narrow widths. Sample options retain accessible names and selected state. Body copy remains at least 16px; secondary controls and footer copy use 14px. Technical text moves behind disclosure rather than shrinking to fit.

The Accuracy view uses one comparison table with named methods, dataset sizes and explicit error units. A worked example explains percentage points; evaluation configuration lives in a disclosure. Headings name the content directly.

## Public release

The public preview hides upload and local research controls. It has no sample-review form. Dataset review remains a separate local development workflow for 524 records. About lives in the same persistent shell. Product navigation contains no links to the separate local research tools. Mode changes retain uploaded images and the most recent completed upload result; Accuracy and About do not mutate analysis state.
