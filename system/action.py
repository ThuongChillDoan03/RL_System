def _player_action_(act, p_idx):
    if act == 0:
        print(f'Người chơi {p_idx+1} kết thúc lượt:', act)
    elif act in range(1:13):
        id_action = act-1
        id_card_normal = get_id_card_normal in lv(lv1, lv2, lv3)
        