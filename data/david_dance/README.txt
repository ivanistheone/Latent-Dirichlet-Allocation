

Mon 22 Aug 2011 15:51:49 EDT



==== Data ====

Roughly 200 answers to a questionnaire about partnering in dance couples.
After removing stop-words like "is", "he" and "the" we are left with 2145 words,
which is not a lot.



==== Preprocessing ====

1/ Preparing the word stream
    ./csv_to_low.py
    This splits the csv file into `ids.txt` and `docs.txt`.
    Removes stopwords and stems words.


2/ Building a vocabulary
    ./makevocab.py -i docs.txt -o tmp_v.txt --minoccur 4
    We have dropped words that occur less than 4 times in the answers.
    The result is the file `vocab.txt` which has 128 different terms.



    I have tried with stemming and non-stemming...
    Also cutoff minoccur 3 and minoccur 2
    and more topics seems to find better topics.




==== Running LDA ====

I asked the algorithm to produce clusters of 5, 10, 15 and 20 topics.
Each time I adjusted the beta parameter, so that topics use different words
as much as possible and the alpha parameter so that each user answer is 
characterized by a few topics.

The results consist of the most likely words in each topic.


=== 

probook:david_dance ivan$ ../../run.py --docs docs.txt --vocab vocab.txt --iter 800 --save_probs --print_topics 14 --rundirs_root /CurrentPorjects/LatentDirichletAllocation/data/david_dance/ --alpha 0.01 --beta 0.005 --numT 29 --seed 22
    ...
    itr = 798
    itr = 799
    2011-08-25 19:06:39,619 : INFO : Perplexity: 46.817115
    2011-08-25 19:06:39,622 : INFO : Phi sparseness. center=19, width=5
    2011-08-25 19:06:39,626 : INFO : Theta sparseness. center=6, width=6
    2011-08-25 19:06:39,627 : INFO : Done saving output.json

    (one topic per line)

    cue, direct, non, verbal, react, receiv, say, delay, split, time, abl, hand, inform, later
    center, common, similar, let, bodi, movement, social, hand, say, forc, line, level, meet, energi
    time, tri, awar, choreographi, understand, music, develop, teach, teacher, person, partnership, requir, learn, inform
    feel, moment, share, enjoy, qualiti, techniqu, music, experi, emot, perfect, danc, level, male, dancer
    movement, essenti, step, point, relationship, visual, function, facilit, speak, systemat, balanc, order, enhanc, momentum
    ballet, element, dancer, ballroom, polit, play, choreograph, classic, form, modern, enhanc, add, influenc, connect
    danc, connect, touch, time, space, bring, woman, reduc, repetit, synchron, basic, anxieti, mean, nice
    resist, bodi, relat, partnership, weight, person, understand, follow, speed, defin, joint, knowledg, mirror, express
    contact, improv, help, practic, sensit, dancer, train, touch, awar, improvis, eye, close, exercis, basic
    connect, trust, support, direct, chemistri, communic, space, eas, nice, dialogu, mutual, perform, respect, agre
    contact, danc, visual, depend, express, physic, relat, feel, choreographi, style, woman, look, relationship, want
    breath, level, anxieti, check, especi, coupl, belli, eachoth, heartbeat, resolv, salsa, chest, fit, present
    time, lead, lift, import, chang, support, big, develop, simpli, materi, women, profession, maintain, easier
    power, tool, sensori, memori, strength, concentr, engag, left, motor, pick, quick, contact, common, natur
    breath, rhythm, import, think, flow, necessari, equal, shape, movement, natur, speed, idea, puls, rythm
    movement, term, physic, initi, turn, effort, momentum, impuls, happen, american, basi, imposs, intent, sequenc
    balanc, weight, lift, support, posit, lean, want, pose, stretch, connect, floor, involv, partner, contact
    point, centr, graviti, bodi, share, meet, general, blend, import, ballroom, contact, improvis, function, convey
    weight, transfer, forward, touch, follow, releas, axi, breath, chest, later, backward, foot, signal, need
    perform, differ, question, contemporari, mean, follow, abil, concentr, recogn, unchoreograph, term, aspect, partner, contact
    depend, touch, lead, connect, requir, hold, choreographi, natur, stori, strong, cours, core, moment, essenti
    contact, visual, exercis, element, form, open, aesthet, import, ballroom, physic, approach, evolv, simpli, necessarili
    danc, tango, think, import, ideal, ballroom, latin, achiev, control, type, posit, match, championship, chi
    need, understand, pull, push, sure, thing, eye, term, carri, reason, case, awkward, convers, freedom
    contact, weight, touch, share, hand, possibl, improvis, explor, lot, sens, clear, offer, start, visual
    social, communic, awar, similar, competit, focus, physic, experi, abil, aspect, context, communiti, inner, manifest
    movement, bodi, creat, help, explor, surfac, dancer, order, fit, awar, mechan, shoulder, simultan, effici
    feel, bodi, danc, person, abl, listen, share, languag, mind, ground, energi, materi, yield, predict
    need, bodi, style, movement, communic, tri, chang, question, abl, lean, fall, present, catch, coordin


which makes me think what it will find with the same parameters but with no stemming:

    2011-08-25 19:18:16,295 : INFO : Perplexity: 52.728065
    2011-08-25 19:18:16,298 : INFO : Phi sparseness. center=28, width=23
    2011-08-25 19:18:16,304 : INFO : Theta sparseness. center=6, width=6
    2011-08-25 19:18:16,305 : INFO : Done saving output.json

    movement, important, bodies, lead, partners, movements, steps, visual, rhythm, touching, kind, story, timing, natural
    breathing, important, breath, basic, anxiety, speed, levels, steps, breaths, control, eachother, heartbeat, match, reduce
    connection, feel, music, moment, communication, ease, experience, ability, emotion, mutual, direction, resist, slightly, willingness
    required, touch, movements, need, transfer, lead, sure, terms, impulse, american, eventually, risk, style, follow
    partner, depends, dance, need, movement, choreography, visual, physical, relationship, feel, style, dancing, performance, relate
    partner, weight, touch, partnership, essential, sharing, aspects, moves, similar, aware, floor, performance, eye, beginners
    dance, social, connection, tango, element, center, ballet, dancer, modern, connect, perform, form, political, adds
    dance, dancing, moving, makes, feel, person, times, memory, mindful, trying, ways, direction, carried, central
    ballet, time, dancers, choreography, change, able, classical, course, power, arts, lady, martial, connection, professional
    weight, improve, practice, partner, lot, axis, hands, helps, catching, falling, foot, forward, laterally, simple
    simply, ballroom, physical, lead, partnering, form, dancers, exercise, line, vocabulary, question, communicate, dances, aesthetic
    ballroom, woman, partner, contemporary, stage, dancing, political, hand, want, audience, danced, enjoy, feels, focus
    resistance, weight, cue, body, partner, partners, partnership, say, related, directly, definately, knowledge, meet, previously
    body, help, try, surfaces, movement, lift, ballroom, hands, bodies, moment, dancer, support, aware, chest
    point, shared, important, improv, supported, weight, support, maybe, meeting, bodies, centre, sharing, strength, balance
    trust, movement, shared, chemistry, space, big, effort, lifts, works, dancers, awareness, practice, dialogue, timing
    create, eyes, training, strong, improv, energy, fit, case, closed, reason, helps, moments, powerful, looking
    touch, communication, understanding, expression, pull, push, release, time, brings, happen, visual, create, woman, related
    moving, think, style, dance, tango, chest, depending, latin, achieve, breathe, chi, doesn, follows, hard
    level, dancer, perfect, connected, present, male, turns, women, belly, check, closer, effect, engaged, ideally
    ways, touch, explore, possibilities, improvisation, limits, dances, impulse, experienced, head, offers, shapes, tai, contact
    visual, elements, improvisation, difference, think, leading, styles, improve, play, teaching, important, question, generally, present
    leaning, change, time, listening, holding, lifting, express, core, hands, connected, balanced, keeps, persons, position
    common, center, partners, partnering, person, centre, touch, gravity, similar, order, point, energy, depending, share
    understanding, technique, joints, question, allowing, exploration, follower, frame, interesting, leader, relation, quality, react, means
    breath, rhythm, going, movements, balance, flow, partner, body, making, thinking, allow, definitely, equally, pulse
    body, feel, material, language, able, especially, listen, recognize, yielding, dance, contact, partner, weight, touch
    common, cues, non, verbal, sensory, term, tool, visual, powerful, basis, grounded, motor, pick, learning
    awareness, body, important, depends, exercises, sensitivity, need, context, open, things, social, information, let, necessarily



So it is not clear to me whether stemming helps...


    


=== 5, 10, 15 and 20 topics ===

Earlier experiments attempted with smaller number of topics -- but
none seemed to provide coherent topics.

../../run.py --docs docs.txt --vocab vocab.txt --numT 5 --iter 800 --save_probs --print_topics 20 --rundirs_root /CurrentPorjects/LatentDirichletAllocation/data/david_dance/ --alpha 0.05 --beta 0.01
../../run.py --docs docs.txt --vocab vocab.txt --numT 10 --iter 800 --save_probs --print_topics 20 --rundirs_root /CurrentPorjects/LatentDirichletAllocation/data/david_dance/ --alpha 0.02 --beta 0.01
../../run.py --docs docs.txt --vocab vocab.txt --numT 15 --iter 800 --save_probs --print_topics 20 --rundirs_root /CurrentPorjects/LatentDirichletAllocation/data/david_dance/ --alpha 0.01 --beta 0.01
 ../../run.py --docs docs.txt --vocab vocab.txt --numT 20 --iter 800 --save_probs --print_topics 20 --rundirs_root /CurrentPorjects/LatentDirichletAllocation/data/david_dance/ --alpha 0.01 --beta 0.01







==== Topic labelling  ====


 - David must select one of the above choices of # of topics as the most informative.
 - provide short labels (tags) for each topic



====  Classification ====


answer = one line in the .xls
person = sum of all the answers for one person
question = sum of the answers of all people for that question

 - I can then label each answer and and each person according to which topics it contains.



