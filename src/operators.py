def get_forward_operator(cfg):

    selected_operator = cfg["selected_operator"]

    if selected_operator == 'identity':
        def forward_operator(x):
            return x
        return forward_operator
    