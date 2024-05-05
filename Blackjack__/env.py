import numpy as np
def initEnv():

    env_state = np.full(207,0)
    env_state[:9] = 32
    env_state[9] = 128
    env_state[10] = 20000
    env_state[11:17] = 1000

    for a_ in range(7):
        env_state[29 + a_*2] = 1

    return env_state


def getAgentState(env_state):

    P_state = np.full(158,0) 
    P_id = int((env_state[43]%14)//2)
    P1_id = int(env_state[43]%14)    # Bộ nhỏ của mỗi người
    for i in range(7):
        if (P_id+i) <= 6:
            P_state[i] = env_state[10+P_id+i]
        elif (P_id+i) > 6:
            P_state[i] = env_state[3+P_id+i]
    for lct in range(7):
        if (P_id+lct) <= 6:
            P_state[(7+20*lct):(7+20*(lct+1))] = env_state[(44+(P_id+lct)*22):(42+(P_id+lct+1)*22)]
        elif (P_id+lct) > 6:
            P_state[(7+20*lct):(7+20*(lct+1))] = env_state[(44+(P_id+lct-7)*22):(42+(P_id+lct-6)*22)]

    if P1_id <=1:
        P_state[147] = 1
    if P1_id >= 2:
        P_state[147] = env_state[15+P1_id]
    P_state[148] = env_state[198]    # số ván đã chơi
    for dtb in range(7):               # số ván nhỏ win
        if (P_id+i) <= 6:
            P_state[149+dtb] = env_state[199+P_id+dtb]
        elif (P_id+1) > 6:
            P_state[149+dtb] = env_state[192+P_id+dtb]
    P_state[157] = P_id
    return P_state.astype(np.float64)


def getValidActions(P_state):
    
    Valid_Actions_return = np.full(9,0)
    Check_place_a_bet = P_state[147]
    Check_coin_player = P_state[:7]
    Card_on_hand = P_state[7:27]
    Card_on_hand_1 = P_state[7:17]
    Card_on_hand_2 = P_state[17:27]
    Sum_number_of_card = np.sum(Card_on_hand)

    if Check_place_a_bet == 0:
        if Check_coin_player[0] >= 100:
            Valid_Actions_return[0:4] = 1
        elif Check_coin_player[0]<100 and Check_coin_player[0]>=50:
            Valid_Actions_return[0:3] = 1
        elif Check_coin_player[0]<50 and Check_coin_player[0]>=20:
            Valid_Actions_return[0:2] = 1
        elif Check_coin_player[0]<20 and Check_coin_player[0]>=10:
            Valid_Actions_return[0] = 1
        elif Check_coin_player[0]<10:
            Valid_Actions_return[4] = 1
    if Check_place_a_bet == 1:
        Valid_Actions_return[8] = 1                   
    if Check_place_a_bet != 0 and Check_place_a_bet != 1:
        check_place = 0
        for s_ in range(len(Card_on_hand_2)):
            if Card_on_hand_2[s_] != 0:
                check_place += 1
        
        card_other_0 = 0
        for run in range(len(Card_on_hand)):            # 20 lá trên tay
            if Card_on_hand[run] != 0:
                card_other_0 += 1

        if card_other_0==1 and Sum_number_of_card==2:
            if check_place == 0:
                if Check_coin_player[0] >= Check_place_a_bet:
                    Valid_Actions_return[4:8] = 1
            Valid_Actions_return[4:6] = 1
        if card_other_0!=1 and Sum_number_of_card==2:
            Valid_Actions_return[4:6] = 1
            if check_place == 0:
                if Check_coin_player[0] >= Check_place_a_bet:
                    Valid_Actions_return[4:7] = 1
        if Sum_number_of_card >= 3 and check_place == 0:
            Valid_Actions_return[4:6] = 1
        if check_place != 0:
            Valid_Actions_return[4:6] = 1
        if Sum_number_of_card == 0:
            Valid_Actions_return[4] = 1

    return Valid_Actions_return.astype(np.int64)



def stepEnv(action,env_state):
    
    P_player = int((env_state[43]%14)//2)
    status_player = env_state[29:43]
    card_on_hand = env_state[(44+P_player*22):(32+(P_player+1)*22)] 
    card_on_hand_2 = env_state[(32+(P_player+1)*22):(42+(P_player+1)*22)]
    point_card = env_state[42+(P_player+1)*22]
    point_card_2 = env_state[43+(P_player+1)*22]
    remaining = np.sum(env_state[:10])
    check_b = 1
    if remaining <= 8:
        env_state[:10] = [32,32,32,32,32,32,32,32,32,128]

    if status_player[int(env_state[43]%14)] == 0:
        check_b = 0
        env_state[43] += 1
    if status_player[int(env_state[43]%14)]==1 and check_b == 1:  
        if P_player >= 1: 
            if action == 0:
                env_state[15+2*P_player] += 10 #tiền đặt
                env_state[10+P_player] -= 10   #tiền bị trừ đi

                weighted_random = np.array(env_state[:10])
                for i_ in range(2):
                    rate_random = weighted_random/np.sum(weighted_random)
                    choice_place = np.random.choice(np.arange(len(weighted_random)), p=rate_random)
                    weighted_random[choice_place] -= 1
                    card_on_hand[choice_place] += 1
                    if choice_place >= 1:
                        point_card = point_card + choice_place + 1
                    if choice_place == 0:
                        if point_card >= 11:
                            point_card = point_card + 1
                        if point_card <= 10:
                            point_card = point_card + 11

                    env_state[(44+P_player*22):(32+(P_player+1)*22)] = card_on_hand
                    env_state[42+(P_player+1)*22] = point_card
                    env_state[:10] = weighted_random

                env_state[43] += 1
            if action == 1:
                env_state[15+2*P_player] += 20 #tiền đặt
                env_state[10+P_player] -= 20   #tiền bị trừ đi

                weighted_random = np.array(env_state[:10])
                for i_ in range(2):
                    rate_random = weighted_random/np.sum(weighted_random)
                    choice_place = np.random.choice(np.arange(len(weighted_random)), p=rate_random)
                    weighted_random[choice_place] -= 1
                    card_on_hand[choice_place] += 1
                    if choice_place >= 1:
                        point_card = point_card + choice_place + 1
                    if choice_place == 0:
                        if point_card >= 11:
                            point_card = point_card + 1
                        if point_card <= 10:
                            point_card = point_card + 11
                            
                    env_state[(44+P_player*22):(32+(P_player+1)*22)] = card_on_hand
                    env_state[42+(P_player+1)*22] = point_card
                    env_state[:10] = weighted_random

                env_state[43] += 1

            if action == 2:
                env_state[15+2*P_player] += 50 #tiền đặt
                env_state[10+P_player] -= 50   #tiền bị trừ đi

                weighted_random = np.array(env_state[:10])
                for i_ in range(2):
                    rate_random = weighted_random/np.sum(weighted_random)
                    choice_place = np.random.choice(np.arange(len(weighted_random)), p=rate_random)
                    weighted_random[choice_place] -= 1
                    card_on_hand[choice_place] += 1
                    if choice_place >= 1:
                        point_card = point_card + choice_place + 1
                    if choice_place == 0:
                        if point_card >= 11:
                            point_card = point_card + 1
                        if point_card <= 10:
                            point_card = point_card + 11
                            
                    env_state[(44+P_player*22):(32+(P_player+1)*22)] = card_on_hand
                    env_state[42+(P_player+1)*22] = point_card
                    env_state[:10] = weighted_random

                env_state[43] += 1
            if action == 3:
                env_state[15+2*P_player] += 100 #tiền đặt
                env_state[10+P_player] -= 100   #tiền bị trừ đi

                weighted_random = np.array(env_state[:10])
                for i_ in range(2):
                    rate_random = weighted_random/np.sum(weighted_random)
                    choice_place = np.random.choice(np.arange(len(weighted_random)), p=rate_random)
                    weighted_random[choice_place] -= 1
                    card_on_hand[choice_place] += 1
                    if choice_place >= 1:
                        point_card = point_card + choice_place + 1
                    if choice_place == 0:
                        if point_card >= 11:
                            point_card = point_card + 1
                        if point_card <= 10:
                            point_card = point_card + 11
                            
                    env_state[(44+P_player*22):(32+(P_player+1)*22)] = card_on_hand
                    env_state[42+(P_player+1)*22] = point_card
                    env_state[:10] = weighted_random

                env_state[43] += 1
            if action == 4:
                status_player[int(env_state[43]%14)] == 0
                env_state[29+(env_state[43]%14)] = 0
                env_state[43] += 1
            if action == 5:
                if env_state[43]%2 == 0:
                    weighted_random = np.array(env_state[:10])
                    rate_random = weighted_random/np.sum(weighted_random)
                    choice_place = np.random.choice(np.arange(len(weighted_random)), p=rate_random)
                    weighted_random[choice_place] -= 1
                    card_on_hand[choice_place] += 1
      
                    check_card_other_1 = card_on_hand[1:]
                    point_temporary = 0
                    for is_ in range(len(check_card_other_1)):
                        if check_card_other_1[is_] != 0:
                            point_temporary += check_card_other_1[is_]*(is_+2)
                    if point_temporary >= 11:
                        point_temporary += card_on_hand[0]
                    elif point_temporary <= 10:
                        if card_on_hand[0] == 1:
                            point_temporary += 11
                        if card_on_hand[0] != 1:
                            point_temporary += 11 + (card_on_hand[0] - 1)
                    

                    env_state[(44+P_player*22):(32+(P_player+1)*22)] = card_on_hand
                    env_state[42+(P_player+1)*22] = point_temporary
                    env_state[:10] = weighted_random
                if env_state[43]%2 != 0:
                    weighted_random = np.array(env_state[:10])
                    rate_random = weighted_random/np.sum(weighted_random)
                    choice_place = np.random.choice(np.arange(len(weighted_random)), p=rate_random)
                    weighted_random[choice_place] -= 1
                    card_on_hand_2[choice_place] += 1

                    check_card_other_2 = card_on_hand_2[1:]
                    point_temporary_1 = 0
                    for is_ in range(len(check_card_other_2)):
                        if check_card_other_2[is_] != 0:
                            point_temporary_1 += check_card_other_2[is_]*(is_+2)
                    if point_temporary_1 >= 11:
                        point_temporary_1 += card_on_hand_2[0]
                    elif point_temporary_1 <= 10:
                        if card_on_hand_2[0] == 1:
                            point_temporary_1 += 11
                        if card_on_hand_2[0] != 1:
                            point_temporary_1 += 11 + (card_on_hand_2[0] - 1)

                    env_state[(32+(P_player+1)*22):(42+(P_player+1)*22)] = card_on_hand_2
                    env_state[43+(P_player+1)*22] = point_temporary_1
                    env_state[:10] = weighted_random

                env_state[43] += 1
            if action == 6:
                env_state[15+2*P_player] *= 2
                env_state[10+P_player] -= int(env_state[15+2*P_player]/2)

                weighted_random = np.array(env_state[:10])
                rate_random = weighted_random/np.sum(weighted_random)
                choice_place = np.random.choice(np.arange(len(weighted_random)), p=rate_random)
                weighted_random[choice_place] -= 1
                card_on_hand[choice_place] += 1
                
                # check_point_other_A = env_state[42+(P_player+1)*22]
                # if choice_place >= 1:
                #     point_card = point_card + choice_place + 1
                # if choice_place == 0:
                #     if check_point_other_A <= 10:
                #         point_card += 11
                #     elif check_point_other_A >= 11:
                #         point_card += 1   

                check_card_other_1 = card_on_hand[1:]
                point_temporary = 0
                for is_ in range(len(check_card_other_1)):
                    if check_card_other_1[is_] != 0:
                        point_temporary += check_card_other_1[is_]*(is_+2)
                if point_temporary >= 11:
                    point_temporary += card_on_hand[0]
                elif point_temporary <= 10:
                    if card_on_hand[0] == 1:
                        point_temporary += 11
                    if card_on_hand[0] != 1:
                        point_temporary += 11 + (card_on_hand[0] - 1)

                env_state[(44+P_player*22):(32+(P_player+1)*22)] = card_on_hand
                env_state[42+(P_player+1)*22] = point_temporary
                env_state[:10] = weighted_random
                status_player[int(env_state[43]%14)] == 0   # dừng turn bộ action!!!
                env_state[29+(env_state[43]%14)] = 0

                env_state[43] += 1        
            if action == 7:
                env_state[16+2*P_player] = env_state[15+2*P_player]
                env_state[10+P_player]-= env_state[16+2*P_player]
                env_state[30+(env_state[43]%14)] = 1

                card_split = env_state[(44+P_player*22):(32+(P_player+1)*22)]
                for s_ in range(len(card_split)):
                    if card_split[s_] == 2:
                        card_split[s_] = 1
                        env_state[32+(P_player+1)*22+s_] = 1
                        env_state[(44+P_player*22):(32+(P_player+1)*22)] = card_split
                if card_split[0] == 1:
                    env_state[42+(P_player+1)*22] = 11
                    env_state[43+(P_player+1)*22] = 11
                if card_split[0] == 0:
                    env_state[42+(P_player+1)*22] = int(env_state[42+(P_player+1)*22]/2)
                    env_state[43+(P_player+1)*22] = env_state[42+(P_player+1)*22] 

                point_card = env_state[42+(P_player+1)*22]
                point_card_2 = env_state[43+(P_player+1)*22]
                weighted_random = np.array(env_state[:10])
                for i_ in range(2):
                    rate_random = weighted_random/np.sum(weighted_random)
                    choice_place = np.random.choice(np.arange(len(weighted_random)), p=rate_random)
                    if i_ == 0:
                        weighted_random[choice_place] -= 1
                        card_on_hand[choice_place] += 1
                        if choice_place == 0:
                            point_card += 11
                        if choice_place != 0:
                            point_card += choice_place + 1
                        env_state[:10] = weighted_random
                        env_state[(44+P_player*22):(32+(P_player+1)*22)] = card_on_hand
                        env_state[42+(P_player+1)*22] = point_card
                    if i_ == 1:
                        weighted_random[choice_place] -= 1
                        card_on_hand_2[choice_place] += 1
                        if choice_place == 0:
                            point_card_2 += 11
                        if choice_place != 0:
                            point_card_2 += choice_place + 1
                        env_state[(32+(P_player+1)*22):(42+(P_player+1)*22)] = card_on_hand_2
                        env_state[:10] = weighted_random
                        env_state[43+(P_player+1)*22] = point_card_2
                env_state[43] += 2
        if P_player == 0:
            if action == 8:     
                asgroup_1 = np.sum(card_on_hand)   # bộ thứ nhất của cái
                test_action = np.sum(status_player[2:])   # tình trạng người chơi còn action ko
                check_a = 1
                if asgroup_1 == 0:     # chưa có lá bài nào hết!!!Bốc 2 lá bài (trong đó có 1 lá úp thêm vào env[206])
                    weighted_random = np.array(env_state[:10])
                    for i_ in range(2):
                        rate_random = weighted_random/np.sum(weighted_random)
                        choice_place = np.random.choice(np.arange(len(weighted_random)), p=rate_random)
                        if i_ == 0:
                            weighted_random[choice_place] -= 1
                            card_on_hand[choice_place] += 1
                            if choice_place == 0:
                                point_card = 11
                            if choice_place != 0:
                                point_card = choice_place + 1
                        if i_ == 1:
                            weighted_random[choice_place] -= 1
                            env_state[206] += choice_place+1
                    if env_state[206] != 1:
                        poin_sum = point_card + env_state[206]
                    if env_state[206] == 1:
                        poin_sum = point_card + 11
                    if poin_sum == 21:
                        env_state[44:54] = [1,0,0,0,0,0,0,0,0,1]
                        env_state[206] = 0
                        env_state[64] = 21
                        env_state[29:31] = 0  
                    if poin_sum != 21:
                        env_state[44:54] = card_on_hand
                        env_state[64] = point_card
                    
                    env_state[:10] = weighted_random
                    check_a = 0
                    env_state[43] += 2
                if asgroup_1 != 0 and check_a == 1:
                    if test_action != 0:                    # Vẫn còn người chơi đang action
                        env_state[43] += 2
                    if test_action == 0:
                        env_state[43+env_state[206]] += 1
                        if env_state[206] != 1:
                            env_state[64] = env_state[64] + env_state[206]
                        if env_state[206] == 1:
                            env_state[64] += 11
                        env_state[206] = 0
                        if env_state[64] > 16:   #ko rút bài nữa
                            status_player[0:2] = 0 
                            env_state[29:31] = 0
                        if 0<env_state[64]<=16:  # rút thêm 1 lá
                            weighted_random = np.array(env_state[:10])
                            rate_random = weighted_random/np.sum(weighted_random)
                            choice_place = np.random.choice(np.arange(len(weighted_random)), p=rate_random)
                            weighted_random[choice_place] -= 1
                            card_on_hand[choice_place] += 1

                            # if choice_place >= 1:
                            #     env_state[64] = env_state[64] + choice_place + 1
                            # if choice_place == 0:
                            #     if env_state[64] >= 11:
                            #         env_state[64] += 1
                            #     elif env_state[64] <= 10:
                            #         env_state[64] += 11

                            check_card_other_1 = card_on_hand[1:]
                            point_temporary = 0
                            for is_ in range(len(check_card_other_1)):
                                if check_card_other_1[is_] != 0:
                                    point_temporary += check_card_other_1[is_]*(is_+2)
                            if point_temporary >= 11:
                                point_temporary += card_on_hand[0]
                            elif point_temporary <= 10:
                                if card_on_hand[0] == 1:
                                    point_temporary += 11
                                if card_on_hand[0] != 1:
                                    point_temporary += 11 + (card_on_hand[0] - 1)

                            env_state[:10] = weighted_random    
                            env_state[44:54] = card_on_hand
                            env_state[64] = point_temporary
                        env_state[43] += 2
                
    #-------------------------------------#####reset_small_game_---------------------------------------#
    check_blj_cai = np.sum(env_state[29:43])
    test_coin = env_state[10:17]
    test_case = []
    for ie in range(len(test_coin)):
        if test_coin[ie] >= 10:
            test_case.append(ie)
    a = test_case[-1]
    if a >= 1:
        if env_state[29] == 0 and env_state[15+a*2] != 0:
            if check_blj_cai == 6:
                status_player = 0
                env_state[29:43] = 0
    
    point_end = env_state[np.array([64,65,86,87,108,109,130,131,152,153,174,175,196,197])]
    for zes in range(14):
        if point_end[zes] >= 21:
            env_state[29+zes] = 0
    check_small_game = np.sum(status_player)
    if check_small_game == 0:
        cardNumbers = []
        for sz_ in range(7):
            cardNumbers.append(np.sum(env_state[(44+sz_*22):(42+(sz_+1)*22)]))
        blackjackPlaces = []
        for nct in range(7):
            if cardNumbers[nct] == 2 and point_end[2*nct] == 21:
                blackjackPlaces.append(nct)

        # check_2 = [] #----------
        # for isd in range(7):
        #         check_2.append(point_end[2*isd] + cardNumbers[isd])
        # check_blackjack = np.array(check_2)   
        # blackjackPlaces = np.where(check_blackjack == 23)[0]     # những người có blj
        asd = [0,1,2,3,4,5,6]                                   # ko có blj
        for rub in blackjackPlaces:
            asd.remove(rub)
        if len(blackjackPlaces) != 0:   #có người có blackjack
            if blackjackPlaces[0] == 0:                             # cái có blj
                for run_ in blackjackPlaces[1:]:
                    env_state[10+run_] += env_state[15+2*run_]
                    env_state[15+2*run_] = 0
                for rub_ in asd:
                    addtional = int(0.5*(env_state[15+2*rub_] + env_state[16+2*rub_]))
                    if env_state[10+rub_] >= addtional:
                        env_state[10] += 3*addtional
                        env_state[10+rub_] -= addtional
                    else:
                        env_state[10] += env_state[15+2*rub_] + env_state[16+2*rub_] + env_state[10+rub_]
                        env_state[10+rub_] = 0
                    env_state[15+2*rub_] = 0
                    env_state[16+2*rub_] = 0   
                    env_state[199] += 1
            if blackjackPlaces[0] != 0:                             # cái ko có blackjack, người chơi có blj  
                for runn in blackjackPlaces:
                    env_state[10] -= int(1.5*(env_state[15+2*runn]))
                    env_state[10+runn] += int(2.5*env_state[15+2*runn])
                    env_state[15+2*runn] = 0
                    env_state[199+runn] += 1 
                
                asd.remove(0)                   # player không có blackjack
                if point_end[0] <= 21:                 # nhà cái ít hơn 22đ
                    for dct in asd:
                        if point_end[2*dct] >= 22:
                            env_state[10] += env_state[15+2*dct]
                            env_state[15+2*dct] = 0
                            env_state[199] += 1
                        if point_end[2*dct+1] >= 22:
                            env_state[10] += env_state[16+2*dct]
                            env_state[16+2*dct] = 0
                            env_state[199] += 1
                        if point_end[2*dct] <= 21:
                            if point_end[2*dct] == env_state[64]:
                                env_state[10+dct] += env_state[15+2*dct]
                                env_state[15+2*dct] = 0
                            elif point_end[2*dct] > env_state[64]:
                                env_state[10+dct] += 2*env_state[15+2*dct]
                                env_state[10] -= env_state[15+2*dct]
                                env_state[15+2*dct] = 0
                                env_state[199+dct] += 1
                            elif point_end[2*dct] < env_state[64]:
                                env_state[10] += env_state[15+2*dct]
                                env_state[15+2*dct] = 0
                                env_state[199] += 1
                        if point_end[2*dct+1] <= 21:
                            if point_end[2*dct+1] == env_state[64]:
                                env_state[10+dct] += env_state[16+2*dct]
                                env_state[16+2*dct] = 0
                            elif point_end[2*dct+1] > env_state[64]:
                                env_state[10+dct] += 2*env_state[16+2*dct]
                                env_state[10] -= env_state[16+2*dct]
                                env_state[16+2*dct] = 0
                                env_state[199+dct] += 1
                            elif point_end[2*dct+1] < env_state[64]:
                                env_state[10] += env_state[16+2*dct]
                                env_state[16+2*dct] = 0
                                if point_end[2*dct+1] > 0:
                                    env_state[199] += 1
                if point_end[0] >= 22:                  # Nhà cái lớn hơn 22đ
                    for dct in asd:
                        if point_end[2*dct] >= 22:
                            env_state[10] += env_state[15+2*dct]
                            env_state[15+2*dct] = 0
                        if point_end[2*dct+1] >= 22:
                            env_state[10] += env_state[16+2*dct]
                            env_state[16+2*dct] = 0
                        if point_end[2*dct] <= 21:
                            env_state[10+dct] += 2*env_state[15+2*dct]
                            env_state[10] -= env_state[15+2*dct]
                            env_state[15+2*dct] = 0
                            env_state[199+dct] += 1
                        if point_end[2*dct+1] <= 21:
                            env_state[10+dct] += 2*env_state[16+2*dct]
                            env_state[10] -= env_state[16+2*dct]
                            env_state[16+2*dct] = 0
                            if point_end[2*dct+1] > 0:
                                env_state[199+dct] += 1
        else:
            asd = [1,2,3,4,5,6]
            if point_end[0] <= 21:                 # nhà cái ít hơn 22đ
                for dct in asd:
                    if point_end[2*dct] >= 22:
                        env_state[10] += env_state[15+2*dct]
                        env_state[15+2*dct] = 0
                        env_state[199] += 1
                    if point_end[2*dct+1] >= 22:
                        env_state[10] += env_state[16+2*dct]
                        env_state[16+2*dct] = 0
                        env_state[199] += 1
                    if point_end[2*dct] <= 21:
                        if point_end[2*dct] == env_state[64]:
                            env_state[10+dct] += env_state[15+2*dct]
                            env_state[15+2*dct] = 0
                        elif point_end[2*dct] > env_state[64]:
                            env_state[10+dct] += 2*env_state[15+2*dct]
                            env_state[10] -= env_state[15+2*dct]
                            env_state[15+2*dct] = 0
                            env_state[199+dct] += 1
                        elif point_end[2*dct] < env_state[64]:
                            env_state[10] += env_state[15+2*dct]
                            env_state[15+2*dct] = 0
                            env_state[199] += 1
                    if point_end[2*dct+1] <= 21:
                        if point_end[2*dct+1] == env_state[64]:
                            env_state[10+dct] += env_state[16+2*dct]
                            env_state[16+2*dct] = 0
                        elif point_end[2*dct+1] > env_state[64]:
                            env_state[10+dct] += 2*env_state[16+2*dct]
                            env_state[10] -= env_state[16+2*dct]
                            env_state[16+2*dct] = 0
                            env_state[199+dct] += 1
                        elif point_end[2*dct+1] < env_state[64]:
                            env_state[10] += env_state[16+2*dct]
                            env_state[16+2*dct] = 0
                            if point_end[2*dct+1] > 0:
                                env_state[199] += 1
            if point_end[0] >= 22:
                for dct in asd:
                    if point_end[2*dct] >= 22:
                        env_state[10] += env_state[15+2*dct]
                        env_state[15+2*dct] = 0
                    if point_end[2*dct+1] >= 22:
                        env_state[10] += env_state[16+2*dct]
                        env_state[16+2*dct] = 0
                    if point_end[2*dct] <= 21:
                        env_state[10+dct] += 2*env_state[15+2*dct]
                        env_state[10] -= env_state[15+2*dct]
                        env_state[15+2*dct] = 0
                        env_state[199+dct] += 1
                    if point_end[2*dct+1] <= 21:
                        env_state[10+dct] += 2*env_state[16+2*dct]
                        env_state[10] -= env_state[16+2*dct]
                        env_state[16+2*dct] = 0
                        if point_end[2*dct+1] > 0:
                            env_state[199+dct] += 1

        env_state[43:198] = 0
        env_state[206] = 0    
        for a_s in range(7):
            env_state[29 + a_s*2] = 1
        env_state[198] += 1
    # print(env_state[:10])
    # print("Số tiền còn lại: ",env_state[10:17])
    # print("Số tiền cược: ", env_state[17:29])
    # print("trạng thái đc chơi: ", env_state[29:43])
    # print("trạng thái bộ nào đang chơi:", env_state[43])
    # print("bài úp:", env_state[206])
    # print("điểm:", point_end)
    # print("Số ván đã chơi:", env_state[198])
    return env_state


def getAgentsize():
    return 7

def checkEnded(env_state):
    pointArr = env_state[10:17]
    if env_state[198] == 5:
        pointArr[0] -= 20000
        for edv in range(6):
            pointArr[edv+1] -= 1000
        maxPoint = np.max(pointArr)
        maxPointPlay = np.where(pointArr==maxPoint)[0]
        # print(maxPointPlay[0])
        if len(maxPointPlay) == 1:
            return maxPointPlay[0] 
        else:
            number_of_win_smallgame = env_state[199:206]
            maxWin_smallgame = np.max(number_of_win_smallgame)
            number_of_win_smallgame_1 = np.array(number_of_win_smallgame)
            maxWin_smallgame_Play = np.where(number_of_win_smallgame_1 == maxWin_smallgame)[0]
            
            return maxWin_smallgame_Play[0]
    else:
        AF_end = 0
        for tex in range(7):
            if pointArr[tex] >= 10:
                AF_end += 1
        if AF_end == 1:
            maxpoin = np.max(pointArr)
            winpoint = np.where(pointArr == maxpoin)[0]
            return winpoint
        else:
            return -1       



def getReward(P_state):
    if P_state[148] != 5:
        scorePoint_Arr = P_state[0:7]
        money_left = 0
        for ted in range(7):
            if scorePoint_Arr[ted] >= 10:
                money_left += 1
        if money_left == 1:
            maxwin = np.max(scorePoint_Arr)
            winnerx = np.where(scorePoint_Arr == maxwin)[0]
            if winnerx == 0:
                return 1
            else:
                return -1
            
        return 0
    else:
        scorePoint_Arr = P_state[0:7]
        maxCoin_pl = np.max(scorePoint_Arr)
        scorePoint_Arr_1 = np.array(scorePoint_Arr)
        if scorePoint_Arr[0] < maxCoin_pl:
            return -1
        else:
            maxCoin_Pl_place = np.where(scorePoint_Arr_1==maxCoin_pl)
            if len(maxCoin_Pl_place) == 1:
                return 1
            else:
                maxNumber_ofSmallWin = P_state[149:156]
                maxWin = np.max(maxNumber_ofSmallWin)
                maxNumber_ofSmallWin_1 = np.array(maxNumber_ofSmallWin)
                if maxNumber_ofSmallWin[0] < maxWin:
                    return -1
                else:    #trường hợp nếu số ván thắng nhỏ bằng max
                    maxWin_player = np.where(maxNumber_ofSmallWin_1==maxWin)   #những người có số ván thắng nhỏ giống nhau
                if len(maxWin_player) == 1:
                    return 1
                else:
                    add = np.full(len(maxWin_player, P_state[157]))
                    setPlayer = maxWin_player + add
                    for place in range(len(setPlayer)):
                        if setPlayer[place] > 6:
                            setPlayer[place] -= 7
                    winner = np.min(setPlayer)
                    if setPlayer[0] == winner:
                        return 1
                    else:
                        return -1


def getStateSize():
   return 158

def run(listAgent,perData):
    env_state = initEnv()
    tempData = [[0],[0],[0],[0],[0],[0],[0]]

    winner = -1
    Id_player = int((env_state[43]%14)//2)
    while env_state[198] <= 50:
        pIdx = int((env_state[43]%14)//2)
        P1_state = getAgentState(env_state)
        list_action = getValidActions(P1_state)
        try:
            action, tempData[pIdx], perData = listAgent[pIdx](P1_state, tempData[pIdx], perData)
        except:
            print(list(env_state))

        if list_action[action] != 1:
            raise Exception('Người chơi trả về action lỗi') 

        stepEnv(action,env_state)
        winner = checkEnded(env_state)
        if winner != -1:
            break
    
    for pIdx in range(7):
        Id_player = pIdx
        P1_state = getAgentState(env_state)
        action, tempData[pIdx], perData = listAgent[pIdx](P1_state, tempData[pIdx], perData)


    return winner, perData


def main(listAgent, num_math, perData):
    if len(listAgent) != 7:
        print('Hệ thống cho đúng 7 người chơi:>>>>')

    numWin = np.full(8,0)
    pIdOrder = np.arange(7)
    for w_ in range(num_math):
        # np.random.shuffle(pIdOrder)
        winner, perData = run([listAgent[pIdOrder[0]], listAgent[pIdOrder[1]], listAgent[pIdOrder[2]], listAgent[pIdOrder[3]], listAgent[pIdOrder[4]], listAgent[pIdOrder[5]], listAgent[pIdOrder[6]]], perData)

        if winner == -1:
            numWin[7] += 1
        else:
            numWin[pIdOrder[winner]] += 1
    
    return numWin, perData


def random_player(P_state, tempData, perData):
    actions = getValidActions(P_state)
    actions = np.where(actions == 1)[0]
    # print(actions)
    action = np.random.choice(actions)
    # print(action)
    return action, tempData, perData

win, _ = main([random_player]*getAgentsize(), 100, [0])
print(win)