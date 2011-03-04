
import munkres
m=munkres.Munkres()



# standard non-confilct
costM1 = [  [1, 2, 3],
            [4, 1, 45],
            [9, 9, 1]  ]

map = m.compute(costM1)
assert map == [(0, 0), (1, 1), (2, 2)]


# more workers than tasks
costM2 = [  [1, 2, 3],
            [4, 1, 45],
            [9, 9, 1],
            [9, 9, 0.5 ]]

map = m.compute(costM2)
assert map ==  [(0, 0), (1, 1), (3, 2)]
# i.e. third worker (id=2) is not assigned to any task




# more tasks than workers
costM3 = [  [1, 2, 3,   0.5],
            [4, 1, 45,  20],
            [9, 9, 1,   10]]

map = m.compute(costM3)
assert map ==  [(0, 3), (1, 1), (2, 2)]
# i.e. nobody will do task 0


