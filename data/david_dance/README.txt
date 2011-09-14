

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



Topic label                         Top words in topic
0  <center>                            ['center', 'common', 'similar', 'graviti', 'centr', 'effort', 'experienc']
1  <body language>                     ['bodi', 'feel', 'abl', 'listen', 'danc', 'person', 'mind']
2  <?>                                 ['danc', 'feel', 'share', 'think', 'experi', 'techniqu', 'style']
3  <visual contact and expression>     ['visual', 'contact', 'danc', 'physic', 'look', 'essenti', 'stage']
4  <?>                                 ['danc', 'connect', 'perform', 'touch', 'import', 'concentr', 'creat']
5  <initiation, action/reaction>       ['resist', 'touch', 'cue', 'relat', 'direct', 'hold', 'speed']
6  <supporting each other>             ['connect', 'communic', 'time', 'level', 'breath', 'hold', 'anxieti']
7  <sharing emotions/sensations>       ['moment', 'feel', 'enjoy', 'emot', 'perfect', 'share', 'music']
8  <leading>                           ['ballroom', 'dancer', 'lead', 'level', 'import', 'abil', 'materi']
9  <teaching to partner>               ['contact', 'time', 'develop', 'lead', 'communic', 'choreographi', 'teach']
10 <equality in level>                 ['trust', 'movement', 'direct', 'space', 'chemistri', 'share', 'nice']
11 <center 2>                          ['point', 'hand', 'bodi', 'share', 'balanc', 'centr', 'lean']
12 <weight bearing>                    ['time', 'lift', 'improv', 'support', 'tri', 'chang', 'movement']
13 <rhythm in partnering>              ['partnership', 'music', 'import', 'person', 'rhythm', 'defin', 'natur']
14 <initiation, action/reaction 2>     ['movement', 'non', 'verbal', 'cue', 'power', 'sensori', 'term']
15 <breath and partnering>             ['weight', 'support', 'transfer', 'follow', 'chest', 'forward', 'breath']
16 <breath and rhythm>                 ['breath', 'rhythm', 'danc', 'movement', 'think', 'import', 'tango']
17 <learning from other styles>        ['element', 'polit', 'danc', 'center', 'ballet', 'practic', 'enhanc']
18 <good contact depends on...>        ['depend', 'awar', 'social', 'touch', 'physic', 'choreographi', 'context']
19 <choreography requirements>         ['requir', 'connect', 'dancer', 'lead', 'happen', 'ballet', 'classic']
20 <training, exercises>               ['contact', 'improv', 'exercis', 'train', 'form', 'social', 'open']
21 <movement mechanics>                ['need', 'movement', 'bodi', 'awar', 'understand', 'pull', 'thing']
22 <exploration>                       ['touch', 'weight', 'explor', 'share', 'possibl', 'floor', 'posit']
23 <communication>                     ['bodi', 'understand', 'follow', 'movement', 'relat', 'question', 'allow']
24 <?>                                 ['depend', 'need', 'express', 'relat', 'choreographi', 'feel', 'style']
25 <style differences>                 ['ballroom', 'style', 'woman', 'element', 'tango', 'differ', 'ballet']
26 <exploration 2>                     ['bodi', 'movement', 'help', 'creat', 'surfac', 'explor', 'contact']
27 <projection through relationship>   ['person', 'centr', 'relationship', 'understand', 'energi', 'audienc', 'function']
28 <eye contact>                       ['contact', 'help', 'eye', 'sensit', 'close', 'weight', 'check']



==== Topic labelling  ====


 - David must select one of the above choices of # of topics as the most informative.
 - provide short labels (tags) for each topic

 - DONE (see above)




====  Classification ====


answer = one line in the .xls
person = sum of all the answers for one person
question = sum of the answers of all people for that question

 - I can then label each answer and and each person according to which topics it contains.






==== PART 1  ====

Treating each answer as a document


Who Que First words...   Top topics (prob)
=== === ================ ================================================================================
1   1   Non-verbal cu... initiation, action/reaction 2(0.37), body language(0.31), initiation, action/reaction(0.12), movement mechanics(0.12)
1   2   Being connect... supporting each other(0.55), body language(0.22), initiation, action/reaction 2(0.16)
1   3   The psychogic... rhythm in partnering(0.49), supporting each other(0.25), good contact depends on...(0.25)
1   4   Connection wi... breath and rhythm(0.75), supporting each other(0.22)
1   5   Yes, breath a... body language(0.54), breath and rhythm(0.22), choreography requirements(0.11), eye contact(0.11)
1   6   Yes, touch is... initiation, action/reaction 2(0.67), ?(0.31)
1   7   I take or giv... breath and partnering(0.98)
1   8   Since verbal ... initiation, action/reaction 2(0.97)
1   9   The common ce... projection through relationship(0.62), movement mechanics(0.36)
1   10  Breathing int... body language(0.47), breath and rhythm(0.47)
2   1   40's female, ... ?(0.26), leading(0.26), style differences(0.26), choreography requirements(0.13)
2   2   No tricks... ... breath and rhythm(0.91)
2   3   Clarity, brea... weight bearing(0.33), breath and partnering(0.24), eye contact(0.24), visual contact and expression(0.08), leading(0.08)
2   4   Not sure how ... style differences(0.49), training, exercises(0.27), center 2(0.16)
2   5   Yes. as above... breath and rhythm(0.75), center(0.11), ?(0.11)
2   6   Yes. One need... teaching to partner(0.48), center 2(0.32), ?(0.16)
2   7   Sometimes it ... learning from other styles(0.41), movement mechanics(0.41), visual contact and expression(0.14)
2   8   It can be the... ?(0.61), movement mechanics(0.31)
2   9   I don't use t... center 2(0.65), learning from other styles(0.32)
2   10  Simple leans ... center 2(0.48), leading(0.24), eye contact(0.24)
3   1   I am a male P... communication(0.51), ?(0.33), leading(0.15)
3   2   Of Course a g... initiation, action/reaction(0.37), communication(0.29), rhythm in partnering(0.11), choreography requirements(0.11), exploration 2(0.11)
3   3   In partner wo... communication(0.54), initiation, action/reaction(0.39)
3   4   Like i mentio... visual contact and expression(0.65), rhythm in partnering(0.24), initiation, action/reaction(0.08)
3   5   Definately! I... leading(0.39), rhythm in partnering(0.39), sharing emotions/sensations(0.20)
3   6   Definately! L... initiation, action/reaction(0.50), rhythm in partnering(0.16), exploration 2(0.07)
3   7   Weight in par... breath and partnering(0.43), initiation, action/reaction(0.30), body language(0.26)
3   8   When we use v... visual contact and expression(0.30), initiation, action/reaction(0.30), communication(0.25), teaching to partner(0.15)
3   9   For me the wo... center(0.43), initiation, action/reaction(0.37)
3   10  Leaning towar... initiation, action/reaction(0.60), center 2(0.24), visual contact and expression(0.12)
4   1   Responsibilit... learning from other styles(0.57), weight bearing(0.19), exploration 2(0.19)
4   2   Practise and ... 
4   3   Know how your... 
4   4   Most dance is... learning from other styles(0.68), breath and rhythm(0.30)
4   5   Breath and ry... breath and rhythm(0.95)
4   6   You must have... visual contact and expression(0.47), body language(0.24), learning from other styles(0.24)
4   7   When supporti... exploration(0.48), weight bearing(0.36), breath and rhythm(0.12)
4   8   The visual co... visual contact and expression(0.88)
4   9   Common center... center 2(0.36), equality in level(0.27), center(0.18), weight bearing(0.09), choreography requirements(0.09)
4   10  Mentaly going... center 2(0.38), initiation, action/reaction 2(0.38), breath and rhythm(0.19)
5   1   a) aged 48, m... ?(0.52), weight bearing(0.26), supporting each other(0.20)
5   2   I am a direct... equality in level(0.21), teaching to partner(0.16), center 2(0.16), breath and partnering(0.16), movement mechanics(0.16)
5   3   Trust, commit... leading(0.33), weight bearing(0.24), exploration(0.24), equality in level(0.16)
5   4   I think there... style differences(0.24), visual contact and expression(0.21), sharing emotions/sensations(0.18), ?(0.15), equality in level(0.12)
5   5   yes both are ... supporting each other(0.38), weight bearing(0.23), exploration 2(0.23), equality in level(0.08), ?(0.08)
5   6   without it ce... exploration 2(0.43), choreography requirements(0.32), equality in level(0.22)
5   7   in contact im... weight bearing(0.61), exploration 2(0.25)
5   8   it can be ove... exploration(0.57), leading(0.38)
5   9   the point fro... center 2(0.54), supporting each other(0.22), breath and partnering(0.22)
5   10  warming up in... center 2(0.47), eye contact(0.47)
6   1   37, elite per... ?(0.88)
6   2   Good relation... equality in level(0.44), projection through relationship(0.44)
6   3   I think the e... ?(0.78)
6   4   Extremely, th... choreography requirements(0.88)
6   5   Yes, an aware... movement mechanics(0.36), initiation, action/reaction 2(0.24), eye contact(0.24), good contact depends on...(0.12)
6   6   With sensitiv... movement mechanics(0.44), eye contact(0.44)
6   7   It is importa... good contact depends on...(0.61), leading(0.31)
6   8   Understanding... projection through relationship(0.91)
6   9   Trust exercis... weight bearing(0.54), ?(0.43)
6   10  blending of t... center 2(0.44), exploration 2(0.44)
7   1   improv traini... training, exercises(0.88)
7   2   communication... leading(0.38), movement mechanics(0.38), center(0.19)
7   3   Interesting b... ?(0.78)
7   4   definitely, w... communication(0.38), supporting each other(0.19), initiation, action/reaction 2(0.19), breath and partnering(0.19)
7   5   Yes, touch do... exploration(0.78)
7   6   I practice tr... weight bearing(0.55), breath and partnering(0.41)
7   7   quite, it is ... weight bearing(0.70), supporting each other(0.24)
7   8   That is where... center 2(0.88)
7   9   yes, balancin... weight bearing(0.88)
7   10  Different mom... sharing emotions/sensations(0.45), movement mechanics(0.30), exploration 2(0.15), visual contact and expression(0.08)
8   1   A partner-war... exploration 2(0.48), center 2(0.36), exploration(0.12)
8   2   Depends, on w... ?(0.54), visual contact and expression(0.32), rhythm in partnering(0.11)
8   3   Yes, I think ... breath and rhythm(0.58), communication(0.39)
8   4   Touch is a go... breath and rhythm(0.43), ?(0.32), ?(0.22)
8   5   Usually with ... center 2(0.60), rhythm in partnering(0.36)
8   6   Depends on th... visual contact and expression(0.59), sharing emotions/sensations(0.13), ?(0.13), projection through relationship(0.13)
8   7   My first asso... center(0.69), visual contact and expression(0.28)
8   8   going for a h... body language(0.38), weight bearing(0.30), training, exercises(0.23), breath and partnering(0.08)
8   9   Exposed, enga... ?(0.61), leading(0.31)
8   10  Talking smoot... exploration(0.57), projection through relationship(0.38)
9   1   Dialogue, rep... equality in level(0.91)
9   2   I find it ver... exploration 2(0.91)
9   3   Of course, be... body language(0.47), choreography requirements(0.47)
9   4   Release and s... breath and partnering(0.76), choreography requirements(0.19)
9   5   Very much but... ?(0.91)
9   6   Every movemen... center(0.32), choreography requirements(0.32), communication(0.32)
9   7   Yes. Falling ... weight bearing(0.72), good contact depends on...(0.24)
9   8   depends on th... good contact depends on...(0.78)
9   9   every trick t... breath and rhythm(0.70), learning from other styles(0.24)
9   10  looking up to... visual contact and expression(0.88)
10  1   depends on th... exploration 2(0.91)
10  2   the movement ... weight bearing(0.47), rhythm in partnering(0.24), breath and rhythm(0.24)
10  3   depends on th... supporting each other(0.44), good contact depends on...(0.44)
10  4   we called it ... breath and partnering(0.36), equality in level(0.24), sharing emotions/sensations(0.12), choreography requirements(0.12), communication(0.12)
10  5   contact impro... exploration 2(0.96)
10  6   Chemistry or ... equality in level(0.88)
10  7   Holding your ... supporting each other(0.61), body language(0.31)
10  8   It depends on... good contact depends on...(0.88)
10  9   Sure. If this... movement mechanics(0.70), breath and rhythm(0.24)
10  10  Holding your ... supporting each other(0.88)
11  1   maybe contact... training, exercises(0.91)
11  2   Less thinking... breath and rhythm(0.88)
11  3   Yielding into... body language(0.96)
11  4   No hesitation... sharing emotions/sensations(0.78)
11  5   Positive. The... ?(0.80), teaching to partner(0.16)
11  6   Yes, definite... breath and rhythm(0.72), initiation, action/reaction 2(0.12), good contact depends on...(0.12)
11  7   Yes and no. S... ?(0.47), projection through relationship(0.47)
11  8   yielding. lis... body language(0.48), eye contact(0.48)
11  10  i think it is... visual contact and expression(0.55), breath and rhythm(0.41)
12  1   sharing the p... center 2(0.82), training, exercises(0.14)
12  2   Depends on th... good contact depends on...(0.76), sharing emotions/sensations(0.19)
12  3   To be able to... supporting each other(0.41), body language(0.28), projection through relationship(0.28)
12  4   Yes, it is es... projection through relationship(0.61), visual contact and expression(0.31)
12  5   Depends on. A... good contact depends on...(0.49), choreography requirements(0.33), training, exercises(0.16)
12  6   Depends on th... ?(0.78)
12  7   Not too much.... leading(0.88)
12  8   That the part... projection through relationship(0.88)
12  9   Could be. Dep... good contact depends on...(0.78)
12  10  I usually dan... ?(0.82), rhythm in partnering(0.14)
13  1   hmm no not ne... 
13  2   give and take... movement mechanics(0.60), eye contact(0.36)
13  3   physical and ... visual contact and expression(0.95)
13  4   ofcourse. bre... breath and rhythm(0.82), communication(0.14)
13  5   throught the ... center 2(0.57), leading(0.19), learning from other styles(0.19)
13  6   the middle po... supporting each other(0.61), center 2(0.31)
13  7   Connection, c... sharing emotions/sensations(0.54), supporting each other(0.43)
13  8   In terms of c... ?(0.39), movement mechanics(0.39), ?(0.20)
13  9   Connection. B... supporting each other(0.32), equality in level(0.32), body language(0.16), training, exercises(0.16)
13  10  Very importan... leading(0.59), eye contact(0.30), visual contact and expression(0.10)
14  1   Yes, like tai... breath and rhythm(0.42), rhythm in partnering(0.28), choreography requirements(0.28)
14  2   Yes, but the ... choreography requirements(0.55), ?(0.28), breath and partnering(0.14)
14  3   Only when req... choreography requirements(0.64), breath and partnering(0.23), style differences(0.12)
14  4   I feel it is ... sharing emotions/sensations(0.60), leading(0.36)
14  5   The idea of a... center(0.32), supporting each other(0.32), weight bearing(0.16), good contact depends on...(0.16)
14  6   When I demons... breath and partnering(0.33), supporting each other(0.24), weight bearing(0.24), initiation, action/reaction(0.14)
14  7   Sharing the m... sharing emotions/sensations(0.93)
14  8   Try to call b... sharing emotions/sensations(0.64), style differences(0.32)
14  9   Really import... ?(0.44), breath and rhythm(0.44)
14  10  Rhythm yes br... breath and rhythm(0.88)
15  1   Yes, otherwis... ?(0.61), movement mechanics(0.31)
15  2   The shared fe... body language(0.70), center 2(0.24)
15  3   Swing from ri... center(0.44), teaching to partner(0.44)
15  4   sure...u need... movement mechanics(0.88)
15  5   home of conta... training, exercises(0.78)
15  6   no, just prac... learning from other styles(0.44), exploration 2(0.44)
15  7   Championship ... leading(0.29), teaching to partner(0.29), ?(0.23), ?(0.17)
15  8   The man must ... leading(0.76), good contact depends on...(0.19)
15  9   Other styles ... teaching to partner(0.65), ?(0.22), visual contact and expression(0.11)
15  10  Partners must... breath and rhythm(0.76), exploration 2(0.19)
16  1   You cannot ha... ?(0.44), leading(0.44)
16  2   Odd question.... ?(0.44), communication(0.44)
16  3   There are som... visual contact and expression(0.39), supporting each other(0.33), choreography requirements(0.20)
16  4   There is alwa... center(0.61), communication(0.31)
16  5   Very slow mov... teaching to partner(0.54), breath and partnering(0.22), breath and rhythm(0.22)
16  6   Passion, conn... ?(0.44), sharing emotions/sensations(0.44)
16  7   No just aware... exploration 2(0.61), movement mechanics(0.31)
16  8   Non verbal co... initiation, action/reaction 2(0.88)
16  9   In any type o... ?(0.91)
16  10  Yes because i... teaching to partner(0.88)
17  1   I create an a... good contact depends on...(0.44), exploration 2(0.44)
17  2   That you are ... good contact depends on...(0.61), equality in level(0.31)
17  3   Making sure t... movement mechanics(0.70), center 2(0.24)
17  4   Depends on th... ?(0.46), ?(0.46)
17  5   Relax, listen... body language(0.91)
17  6   Listen to eac... body language(0.57), leading(0.19), exploration(0.19)
17  7   My style of d... visual contact and expression(0.54), ?(0.43)
17  8   Yes, we have ... breath and rhythm(0.70), body language(0.24)
17  9   Depends on ch... ?(0.53), initiation, action/reaction(0.36), weight bearing(0.09)
17  10  That depends ... ?(0.86), body language(0.11)
18  1   Again, it dep... ?(0.64), visual contact and expression(0.23), exploration 2(0.12)
18  3   I have not he... ?(0.57), center(0.19), ?(0.19)
18  4   Rehearsal and... training, exercises(0.76), good contact depends on...(0.19)
18  5   Your question... ?(0.56), weight bearing(0.30), projection through relationship(0.08)
18  6   Depends on th... ?(0.55), equality in level(0.41)
18  7   Touch communi... teaching to partner(0.88)
18  8   Chemistry, co... equality in level(0.78)
18  9    Not sure wha... communication(0.61), style differences(0.31)
18  10  Its part of t... breath and rhythm(0.88)
19  1   Why else woul... ?(0.78)
19  2   Can discuss b... center 2(0.78)
19  3   You need to l... movement mechanics(0.47), body language(0.24), leading(0.24)
19  4   A joining of ... projection through relationship(0.78)
19  5   Contact impro... teaching to partner(0.57), eye contact(0.38)
19  6   Excitement, n... equality in level(0.89), weight bearing(0.09)
19  7   Touch, gaze, ... exploration(0.95)
19  8   Trust, shared... equality in level(0.64), ?(0.32)
19  9   Both are powe... ?(0.41), initiation, action/reaction 2(0.41), leading(0.14)
19  10  Both very imp... weight bearing(0.68), sharing emotions/sensations(0.29)
20  1   Yes, it creat... equality in level(0.57), ?(0.38)
20  2   Weight can be... ?(0.60), breath and partnering(0.24), center(0.12)
20  3   In looking at... weight bearing(0.57), visual contact and expression(0.19), initiation, action/reaction(0.19)
20  4   Two or more b... equality in level(0.43), center(0.32), exploration 2(0.22)
20  5   Sharing exper... ?(0.49), weight bearing(0.41), eye contact(0.08)
20  6   The conenecti... ?(0.71), leading(0.27)
20  7   Personally, I... teaching to partner(0.40), movement mechanics(0.23), body language(0.17), choreography requirements(0.17)
20  8   As the origin... learning from other styles(0.32), training, exercises(0.22), eye contact(0.22), communication(0.17)
20  9   Waking up the... eye contact(0.42), body language(0.31), style differences(0.16), good contact depends on...(0.10)
20  10  A common misc... movement mechanics(0.53), training, exercises(0.28), ?(0.11)
20  11  In Contact Im... eye contact(0.33), training, exercises(0.27), initiation, action/reaction(0.22), supporting each other(0.16)
21  1   Jamming in co... training, exercises(0.40), good contact depends on...(0.32), teaching to partner(0.28)
21  2   In Contact Im... learning from other styles(0.53), training, exercises(0.27), center 2(0.18)
21  3   some of the e... exploration 2(0.75), weight bearing(0.15), learning from other styles(0.08)
21  4   Yes breath ca... weight bearing(0.37), eye contact(0.25), breath and rhythm(0.18)
21  5   In contact mo... teaching to partner(0.35), exploration(0.23), communication(0.23), visual contact and expression(0.12)
21  6   contact can h... exploration 2(0.42), training, exercises(0.35), leading(0.21)
21  7   Timing and mu... rhythm in partnering(0.78), teaching to partner(0.20)
21  8   As far as oth... style differences(0.54), ?(0.24), good contact depends on...(0.19)
21  9   Well touch is... exploration(0.78), center(0.20)
21  10  Visual contac... projection through relationship(0.48), visual contact and expression(0.36), choreography requirements(0.12)
22  1   Perhaps some ... training, exercises(0.58), exploration(0.39)
22  2   I think one c... exploration 2(0.65), breath and rhythm(0.15), breath and partnering(0.08), ?(0.08)
22  3   It depends�wh... training, exercises(0.47), leading(0.28), movement mechanics(0.24)
22  4   There are pol... good contact depends on...(0.35), supporting each other(0.23), learning from other styles(0.23), style differences(0.12)
22  5   In ballet the... style differences(0.39), teaching to partner(0.33), learning from other styles(0.20)
22  6   Since contact... training, exercises(0.59), style differences(0.20), center 2(0.10)
22  7   Some dancers/... choreography requirements(0.61), center 2(0.18), teaching to partner(0.12)
22  8   Sometimes per... rhythm in partnering(0.75), ?(0.22)
22  9   In classical ... style differences(0.54), choreography requirements(0.30), movement mechanics(0.15)
22  10  Not really, w... weight bearing(0.42), learning from other styles(0.28), training, exercises(0.28)
22  11  It would be a... choreography requirements(0.45), weight bearing(0.29), learning from other styles(0.17), equality in level(0.08)




==== PART 2  ====

Treating each PERSON as a document
we represent a person by the sum of his/her answers

-- note: the interview persons are treated as different
         people -- since they have different ids.
         if you don't want that -- you have to use
         the same id for their interview questions ....



Who Top topics for this person (prob)
=== ================================================================================
1   initiation, action/reaction 2(0.25), supporting each other(0.12), body language(0.11), breath and rhythm(0.09), breath and partnering(0.09)
2   center 2(0.15), style differences(0.14), breath and rhythm(0.11), learning from other styles(0.08), leading(0.07)
3   initiation, action/reaction(0.31), communication(0.18), rhythm in partnering(0.09), visual contact and expression(0.07), breath and partnering(0.05)
4   learning from other styles(0.23), breath and rhythm(0.21), center 2(0.10), weight bearing(0.09), visual contact and expression(0.08)
5   weight bearing(0.14), equality in level(0.09), ?(0.08), center 2(0.07), exploration 2(0.07)
6   ?(0.18), weight bearing(0.15), projection through relationship(0.11), movement mechanics(0.10), choreography requirements(0.10)
7   weight bearing(0.25), center 2(0.13), breath and partnering(0.10), exploration(0.08), ?(0.06)
8   visual contact and expression(0.14), breath and rhythm(0.12), center 2(0.08), sharing emotions/sensations(0.08), body language(0.07)
9   weight bearing(0.13), exploration(0.11), ?(0.11), choreography requirements(0.10), exploration 2(0.10)
10  exploration 2(0.24), breath and rhythm(0.14), breath and partnering(0.09), good contact depends on...(0.07), weight bearing(0.07)
11  supporting each other(0.29), movement mechanics(0.18), training, exercises(0.14), equality in level(0.09), good contact depends on...(0.09)
12  breath and rhythm(0.26), body language(0.16), ?(0.14), center 2(0.09), visual contact and expression(0.08)
13  good contact depends on...(0.31), projection through relationship(0.16), choreography requirements(0.11), supporting each other(0.09), leading(0.08)
14  ?(0.20), breath and rhythm(0.16), visual contact and expression(0.12), center 2(0.11), movement mechanics(0.11)
15  choreography requirements(0.15), leading(0.13), supporting each other(0.10), breath and partnering(0.09), sharing emotions/sensations(0.08)
16  sharing emotions/sensations(0.35), ?(0.13), body language(0.09), breath and rhythm(0.09), style differences(0.08)
17  movement mechanics(0.38), training, exercises(0.23), learning from other styles(0.13), exploration 2(0.13)
18  teaching to partner(0.23), leading(0.14), ?(0.10), breath and rhythm(0.10), visual contact and expression(0.08)
19  movement mechanics(0.18), good contact depends on...(0.15), ?(0.14), exploration 2(0.11), teaching to partner(0.09)
20  ?(0.49), weight bearing(0.11), ?(0.07), body language(0.06), visual contact and expression(0.06)
21  ?(0.16), equality in level(0.13), teaching to partner(0.12), breath and rhythm(0.12), center 2(0.09)
22  equality in level(0.21), ?(0.19), weight bearing(0.17), ?(0.13), leading(0.06)
23  training, exercises(0.20), eye contact(0.14), movement mechanics(0.13), learning from other styles(0.11), teaching to partner(0.11)
24  exploration 2(0.26), weight bearing(0.13), teaching to partner(0.11), training, exercises(0.09), exploration(0.07)
25  style differences(0.20), exploration 2(0.19), exploration(0.11), ?(0.09), good contact depends on...(0.07)
26  training, exercises(0.28), style differences(0.15), choreography requirements(0.09), learning from other styles(0.07), teaching to partner(0.07)
27  choreography requirements(0.23), weight bearing(0.18), style differences(0.17), rhythm in partnering(0.12), learning from other styles(0.11)




==== PART 3  ====

Treating each QUESTION as a document
we represent a question by the sum of all the 
answers given by the different people.



Que Top topics for this question (prob)
=== ================================================================================
1   initiation, action/reaction 2(0.25), supporting each other(0.12), body language(0.11), breath and rhythm(0.09), breath and partnering(0.09)
2   center 2(0.15), style differences(0.14), breath and rhythm(0.11), learning from other styles(0.08), leading(0.07)
3   initiation, action/reaction(0.31), communication(0.18), rhythm in partnering(0.09), visual contact and expression(0.07), breath and partnering(0.05)
4   learning from other styles(0.23), breath and rhythm(0.21), center 2(0.10), weight bearing(0.09), visual contact and expression(0.08)
5   weight bearing(0.14), equality in level(0.09), ?(0.08), center 2(0.07), exploration 2(0.07)
6   ?(0.18), weight bearing(0.15), projection through relationship(0.11), movement mechanics(0.10), choreography requirements(0.10)
7   weight bearing(0.25), center 2(0.13), breath and partnering(0.10), exploration(0.08), ?(0.06)
8   visual contact and expression(0.14), breath and rhythm(0.12), center 2(0.08), sharing emotions/sensations(0.08), body language(0.07)
9   weight bearing(0.13), exploration(0.11), ?(0.11), choreography requirements(0.10), exploration 2(0.10)
10  exploration 2(0.24), breath and rhythm(0.14), breath and partnering(0.09), good contact depends on...(0.07), weight bearing(0.07)
11  supporting each other(0.29), movement mechanics(0.18), training, exercises(0.14), equality in level(0.09), good contact depends on...(0.09)








