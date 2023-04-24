import networkagent as n

a1 = n.NetworkAgent(None,"A",epsilon=EPSILON,alpha=ALPHA,decay_alpha=ALPHA_DECAY,decay_epsilon=EPSILON_DECAY)
a2 = n.NetworkAgent(None,"B",epsilon=EPSILON,alpha=ALPHA,decay_alpha=ALPHA_DECAY,decay_epsilon=EPSILON_DECAY, is_heuristic=True)
a3 = n.NetworkAgent(None,"C",epsilon=EPSILON,alpha=ALPHA,decay_alpha=ALPHA_DECAY,decay_epsilon=EPSILON_DECAY, is_heuristic=True)
