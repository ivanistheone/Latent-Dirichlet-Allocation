

Mon 22 Aug 2011 15:51:49 EDT



==== Data ====

Roughly 200 answers to a questionnaire about partnering in dance couples.
After removing stop-words like "is", "he" and "the" we are left with 2145 words,
which is not a lot.



==== Preprocessing ====

1/ Preparing the word stream
    ./csv_to_low.py
    This splits the csv file into `ids.txt` and `docs.txt`.

2/ Building a vocabulary
    ./makevocab.py -i docs.txt -o tmp_v.txt --minoccur 4
    We have dropped words that occur less than 4 times in the answers.
    The result is the file `vocab.txt` which has 128 different terms.



==== Running LDA ====

I asked the algorithm to produce clusters of 5, 10, 15 and 20 topics.
Each time I adjusted the beta parameter, so that topics use different words
as much as possible and the alpha parameter so that each user answer is 
characterized by a few topics.

The results consist of the most likely words in each topic.



=== 5 topics ===

../../run.py --docs docs.txt --vocab vocab.txt --numT 5 --iter 800 --save_probs --print_topics 20 --rundirs_root /CurrentPorjects/LatentDirichletAllocation/data/david_dance/ --alpha 0.05 --beta 0.01

topic 0:
contact, dance, depends, visual, dancing, physical, need, choreography, feel, important, ballroom, partnering, movement, think, cues, express, leaning, non, performance, relate

topic 1:
weight, trust, connection, shared, feel, communication, contact, movement, moment, body, support, moving, music, change, dancer, practice, timing, try, level, dancers

topic 2:
movement, common, center, point, breath, breathing, centre, partners, rhythm, able, dance, yes, person, create, body, bodies, connected, order, space, listening

topic 3:
lead, movements, important, connection, touch, yes, required, breath, rhythm, ways, going, steps, bodies, helps, necessary, question, sure, touching, transfer, need

topic 4:
touch, body, partners, resistance, weight, cue, essential, yes, partnership, understanding, awareness, makes, related, person, time, definately, joints, relation, say, term




=== 10 topics ===

../../run.py --docs docs.txt --vocab vocab.txt --numT 10 --iter 800 --save_probs --print_topics 20 --rundirs_root /CurrentPorjects/LatentDirichletAllocation/data/david_dance/ --alpha 0.02 --beta 0.01

movement, feel, music, create, cues, non, verbal, connection, powerful, sensory, term, dancing, awareness, eyes, moments, important, need, effort, depends, expression

breath, rhythm, important, breathing, dance, yes, think, steps, necessary, able, speed, choreography, question, partner, contact, movement, touch, weight, body, depends

point, bodies, ways, movements, movement, balance, going, hands, flow, kind, centre, moving, helps, touching, level, moment, support, sure, partner, contact

dance, dancing, lead, ballroom, partnership, level, dancer, essential, ability, woman, connection, movements, basic, ballet, experience, latin, physical, communication, space, steps

touch, yes, partnering, required, need, lead, movements, transfer, communication, sure, follow, contact, important, essential, awareness, improve, power, terms, question, speed

depends, dance, choreography, need, time, style, important, change, express, leaning, performance, weight, relationship, expression, relate, lifting, feel, partnering, modern, movement

weight, connection, shared, body, feel, dancers, trust, moment, moving, practice, try, timing, support, chest, ease, sharing, supported, works, communication, material

body, resistance, weight, cue, touch, partners, person, understanding, related, makes, joints, pull, relation, say, directly, technique, feel, moment, balance, going

contact, visual, physical, improv, improvisation, look, directly, eye, relationship, makes, relate, improve, partner, dance, movement, touch, weight, body, important, yes

movement, common, partners, center, trust, centre, person, able, chemistry, connected, order, direction, gravity, similar, space, time, technique, energy, physical, timing



=== 15 topics ===

../../run.py --docs docs.txt --vocab vocab.txt --numT 15 --iter 800 --save_probs --print_topics 20 --rundirs_root /CurrentPorjects/LatentDirichletAllocation/data/david_dance/ --alpha 0.01 --beta 0.01

movement, partners, awareness, touch, cues, non, verbal, yes, connection, powerful, sensory, trust, breathing, direction, body, create, order, improve, story, push

depends, ways, chemistry, leaning, express, feel, rhythm, bodies, relationship, kind, choreography, related, partner, contact, dance, movement, touch, weight, body, important

weight, connection, try, moment, practice, chest, dancers, support, works, moving, music, basic, listening, ease, improve, speed, partner, contact, dance, movement

movements, question, sure, transfer, required, term, need, going, balance, power, rhythm, expression, technique, ballet, effort, partner, contact, dance, movement, touch

depends, choreography, need, change, style, dance, time, performance, partnering, expression, relate, modern, necessary, lifting, trust, able, energy, supported, partner, contact

contact, visual, dance, important, dancing, physical, ballroom, think, woman, basic, makes, latin, ballet, steps, experience, eye, look, time, ways, material

body, resistance, touch, weight, cue, understanding, partners, partnership, directly, related, joints, relation, person, space, definately, say, balance, dancing, bodies, makes

movements, lead, helps, moving, touching, flow, follow, hands, contact, movement, partnering, dancers, going, directly, direction, partner, dance, touch, weight, body

touch, yes, communication, contact, improv, connection, partnering, essential, required, supported, terms, change, experience, partner, dance, movement, weight, body, important, depends

lead, dancer, essential, ability, level, moving, partnership, contact, understanding, technique, moment, support, definately, ease, partner, dance, movement, touch, weight, body

shared, trust, movement, timing, body, feel, able, important, breath, level, relationship, weight, space, support, hands, connected, effort, material, time, dancers

common, center, point, centre, movement, bodies, partners, gravity, similar, balance, create, connected, dance, order, say, partner, contact, touch, weight, body

music, movement, feel, pull, moment, push, sharing, create, practice, eye, story, works, person, moving, express, terms, partner, contact, dance, touch

contact, feel, person, body, dancing, communication, eyes, improvisation, important, need, physical, listening, energy, ballroom, steps, moments, relate, improve, partner, dance

breath, rhythm, yes, important, breathing, dance, think, speed, able, going, necessary, partnering, dancers, steps, kind, power, partner, contact, movement, touch




=== 20 topics ===


 ../../run.py --docs docs.txt --vocab vocab.txt --numT 20 --iter 800 --save_probs --print_topics 20 --rundirs_root /CurrentPorjects/LatentDirichletAllocation/data/david_dance/ --alpha 0.01 --beta 0.01

contact, feel, physical, eye, improvisation, dancing, sharing, important, dance, ways, eyes, partner, movement, touch, weight, body, yes, depends, connection, visual

required, transfer, movements, term, touch, need, yes, lead, rhythm, sure, power, style, expression, effort, partner, contact, dance, movement, weight, body

movements, lead, moving, going, touching, follow, hands, flow, ways, balance, breath, point, moment, direction, partner, contact, dance, movement, touch, weight

touch, bodies, important, depends, pull, push, movement, choreography, kind, terms, helps, connection, essential, required, basic, partner, contact, dance, weight, body

point, understanding, question, technique, bodies, sure, centre, support, balance, shared, able, terms, partner, contact, dance, movement, touch, weight, body, important

movement, feel, energy, need, person, music, listening, eyes, moments, body, sharing, partner, contact, dance, touch, weight, important, yes, depends, connection

breath, rhythm, yes, important, breathing, necessary, think, steps, speed, kind, latin, power, partner, contact, dance, movement, touch, weight, body, depends

movement, trust, shared, space, chemistry, order, able, relationship, ease, effort, time, partnership, style, dancer, steps, direction, partner, contact, dance, touch

connection, moment, music, create, feel, communication, speed, able, basic, partner, contact, dance, movement, touch, weight, body, important, yes, depends, visual

common, center, centre, partners, gravity, similar, create, connected, communication, change, effort, relation, partner, contact, dance, movement, touch, weight, body, important

touch, partners, partnering, yes, leaning, connection, person, change, relationship, essential, ways, important, order, body, related, flow, partner, contact, dance, movement

dancers, depends, works, movement, body, experience, story, partner, contact, dance, touch, weight, important, yes, connection, visual, feel, partners, breath, need

weight, practice, try, chest, change, moving, breathing, support, partner, contact, dance, movement, touch, body, important, yes, depends, connection, visual, feel

movement, cues, non, verbal, partners, powerful, sensory, breathing, helps, improve, need, listening, direction, partner, contact, dance, touch, weight, body, important

depends, need, choreography, time, style, express, performance, relate, dance, expression, lifting, modern, feel, necessary, ballroom, improve, partner, contact, movement, touch

timing, dance, feel, shared, weight, trust, able, material, important, level, hands, experience, body, center, moment, connected, supported, breath, partner, contact

body, resistance, weight, cue, partnership, partners, directly, person, essential, makes, related, definately, joints, say, dancing, lead, steps, basic, relation, story

contact, yes, communication, awareness, touch, improv, body, support, lead, ease, supported, center, ability, direction, partner, dance, movement, weight, important, depends

contact, visual, dance, physical, important, think, partnering, ballroom, look, relationship, makes, partner, movement, touch, weight, body, yes, depends, connection, feel

dance, dancing, ballroom, level, dancer, woman, ability, ballet, lead, moving, latin, connection, movements, time, understanding, look, think, ways, partner, contact



==== Furhter steps ====

 - David must select one of the above choices of # of topics as the most informative.
 - I can then label each answer and and each person according to which topics it contains.


Possibly better results can be obtained though stemming, which maps many words
to their common root:
   dance, dancer, dancing, dancers  --->  dance


To be continued....



