# Error analysis

System: TF-IDF + logistic regression, the best system on both splits. The numbers come from the test set. The examples are
real complaints (ODI number in brackets), shortened. The files `errors_tfidf_lr__time.json` and `errors_tfidf_lr__random.json`
hold three random examples for each of the five most frequent mistakes, and `confusion_*.png` shows the confusion matrices.

**About the judgements below.** "Model wrong", "label wrong" or "both defensible" is my own reading of two or three examples per
confusion. It is not a measurement and another reader could decide differently. There is no ground truth better than the label itself.

## What the numbers say

- **Most errors involve `OTHER`.** 64.3 percent of all errors on the random split and 60.6 percent on the time split are a complaint that is
  labelled `OTHER` and predicted as something else, or the other way round. `OTHER` is not one component. It is everything outside
  the 14 named classes plus NHTSA's own `UNKNOWN OR OTHER`, so it is a mix of unrelated problems.
- **The two engine classes are mixed up.** `ENGINE` and `ENGINE AND ENGINE COOLING` are separate NHTSA components that describe
  similar problems. 9.6 percent of the errors on the time split are mix-ups between them (3.1 percent on the random split). If the two classes were merged, time-split accuracy would be 0.740 instead of 0.713.
- **Hotline-style texts are easier.** Complaints that start with "The contact owns ..." (typed by NHTSA staff from a phone call) are 23 percent of the newest data
  and are classified a little better (accuracy 0.737 against 0.705 for the rest).

## The five most frequent mistakes, time split

| Count | True label | Predicted | My judgement |
| --- | --- | --- | --- |
| 420 | ELECTRICAL SYSTEM | OTHER | Both defensible. Model weak |
| 375 | ENGINE AND ENGINE COOLING | ENGINE | Mostly the label rule, drift as well |
| 231 | OTHER | ELECTRICAL SYSTEM | Model defensible, label arguable |
| 186 | OTHER | STRUCTURE | Model wrong on paint, label arguable on doors |
| 180 | OTHER | POWER TRAIN | Model defensible, label arguable |

### 1. ELECTRICAL SYSTEM predicted as OTHER (420)

- [11675004] "The SOS Module has failed three times ... I cannot contact VW for help in case of accident."
- [11742777] "... the radio/head unit ... intermittently fails ... sudden, extremely loud static noise ..."
- [11686354] "Rear back-up camera fails repeatedly. The image ... is garbled ..."

These are electronics that are not part of the car's classic wiring: an emergency call module, an infotainment unit, a camera. The label
`ELECTRICAL SYSTEM` is reasonable. The model has seen few such words in the older training years and falls back to `OTHER`. **Both defensible, and a genuine weakness of the model on newer technology.**

### 2. ENGINE AND ENGINE COOLING predicted as ENGINE (375)

- [11663448] "... a message indicating that the engine was overheating was displayed ... added water to the coolant reservoir."
- [11614476] "... the check engine warning light ... engine overheating ... internal engine coolant leak."
- [11610309] "... notification of NHTSA Campaign Number 23V905000 (Engine and Engine Cooling) ..."

The first two are cooling problems, so the label fits better than `ENGINE`, and the third even contains the component name. Still, the text says "engine" everywhere,
and the training part of the time split has only 0.7 percent of this class (3.5 percent in the test part). **Mostly a label-rule and drift problem: two similar classes whose use changed over time. The model is not clearly wrong.**

### 3. OTHER predicted as ELECTRICAL SYSTEM (231)

- [11763672] "starter had short life 76k on a 2020 ..."
- [11653349] "The horn stopped working less than a year after I bought the car ..."
- [11610655] "... the horn itself will start blowing without command ..."

A starter and a horn are electrical parts. NHTSA filed these under a component outside the 14 named ones. **Label arguable, the model's answer is defensible.**

### 4. OTHER predicted as STRUCTURE (186)

- [11711162] "... the paint is peeling. ... the roof spoiler is completely peeled off ..."
- [11753079] "... the white paint will not stay on the car ... flaking off all over the car."
- [11760889] "... the doors failed to unlock or open. The vehicle was restarted ..."

Peeling paint is cosmetic and is not a structural safety problem, so predicting `STRUCTURE` is **a model error caused by body words (paint, door, roof)**. The third text is a door that does not open, which could be structure, latches or electronics. **Label arguable.**

### 5. OTHER predicted as POWER TRAIN (180)

- [11644083] "Hesitation from a stop. Truck has a delay when pressing on accelerator ..."
- [11738414] "... the vehicle unexpectedly shifted into neutral and shut off ..."
- [11624561] a motorcycle that started by itself in a parking lot.

The first two sound like power train or engine control problems. **The model is defensible, the label arguable.** The third is a motorcycle, which is rare in this data and has no clear place in the list.

## Two more mistakes that only appear in the top five of the random split

- **STRUCTURE predicted as OTHER (226).** [11341770] "The driver's side sliding door ... cannot be opened ... the interior latch was broken", [11590936] "Tailgate randomly opened while in park."
  A door and a tailgate are body parts, so **the label is right and the model is wrong**. The third example ([10700711], "sharp parts and pieces keep falling off the dash") is too vague to judge.
- **OTHER predicted as ENGINE (148).** [11646436] "There is an internal coolant leak ...", [11266005] "vehicle began to smoke from engine bay ... burst into flames."
  These are engine problems. Components outside the 14 named classes, such as fire-related ones, end up in `OTHER`. **The model is defensible, the label depends on how NHTSA grouped it.**

## Summary

Of the seven groups of mistakes above, the model is clearly wrong in two (the paint complaints in group 4 and the doors and tailgate that were predicted as `OTHER`),
and weak on newer electronics in a third (group 1). In the other four, the model's answer is defensible or the label depends on how NHTSA grouped the problem. That is
what label noise looks like in practice, and it is the reason why an accuracy of about 71 to 77 percent is agreement with the NHTSA label and not a measure of absolute correctness.

## Zero-shot GLM-4.5-Flash on the fixed 1,000-item subset

The language model only sees the 15 class names. On these 1,000 items (the same for every system) it scores accuracy 0.565 and macro-F1 0.592, against 0.735 and 0.678
for TF-IDF on the time split. 5 of its 1,000 answers were not a valid class (`TIRE`, `TIRE PRESSURE MONITORING SYSTEM`, `STABILITY CONTROL`, and one answer that repeated the complaint text).
It is correct where TF-IDF is wrong on 9.8 percent of the items and wrong where TF-IDF is right on 26.8 percent. Both are wrong on 16.7 percent.

| Class (items) | F1 GLM | F1 TF-IDF |
| --- | --- | --- |
| ENGINE (191) | 0.45 | 0.82 |
| OTHER (240) | 0.27 | 0.67 |
| VISIBILITY/WIPER (11) | 0.13 | 0.50 |
| FUEL/PROPULSION SYSTEM (38) | 0.56 | 0.73 |
| AIR BAGS (36) | 0.90 | 0.78 |
| FORWARD COLLISION AVOIDANCE (19) | 0.77 | 0.57 |

The two systems fail in different places, which makes sense:

1. **ENGINE predicted as ENGINE AND ENGINE COOLING (107 times, the most frequent mistake).** The model sees two names for the same thing and cannot know which one NHTSA staff use for which complaint.
   It picks `ENGINE AND ENGINE COOLING` for 14.7 percent of all items, when only 3.2 percent are labelled so. A trained model learns this habit from the data, a prompt cannot. **A limit of zero-shot, not a bug of the model.**
2. **OTHER is the weak spot.** The model answers `OTHER` for 7.9 percent of the items while 24 percent are labelled `OTHER`. The real `OTHER` contains components that are not in the list, such as tires, stability control or lighting.
   The model pushes them into the nearest named class (`STRUCTURE` 60 times, `VISIBILITY/WIPER` 28, `FUEL/PROPULSION SYSTEM` 26) or invents a class that is not on the list (the 5 invalid answers).
3. **Where it is better.** On `AIR BAGS` and `FORWARD COLLISION AVOIDANCE`, whose frequency changed a lot over time, the zero-shot model does not suffer from the drift, because it never learned the old class mix. With 36 and 19 items these two numbers are noisy.

My reading, which is not a measurement: the language model understands the complaints well but does not know NHTSA's labelling rules, and this task is mostly about those rules. This is the case where a small trained model has a real advantage.
