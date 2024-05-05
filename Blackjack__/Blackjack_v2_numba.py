import numpy as np
from numba import njit
from numba.typed import List
import random as rd


#-----------------------NOPYTHON FUNCTIONS------------//

@njit()
def initEnv():

   env_state = np.full(207,0)
   env_state[:9] = 32
   env_state[9] = 128
   env_state[10] = 20000
   env_state[11:17] = 1000

   for a_ in range(7):
      env_state[29 + a_*2] = 1

   return env_state

@njit()
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
   for isx in range(7):               # số ván nhỏ win
      if (P_id+i) <= 6:
         P_state[149+isx] = env_state[199+P_id+isx]
      elif (P_id+1) > 6:
         P_state[149+isx] = env_state[192+P_id+isx]
   P_state[157] = P_id
   return P_state.astype(np.float64)

@njit()
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

# Random theo trọng số.
@njit()
def weighted_random(arr_card_on_board):
   """Trả ra index"""
   sum_ = np.sum(arr_card_on_board)
   if sum_ <= 1e-15:
      raise

   rand_card = np.random.uniform(0, sum_)
   for idx_card in range(arr_card_on_board.shape[0]):
      rand_card -= arr_card_on_board[idx_card]
      if rand_card <= 0:
         return idx_card

@njit()
def check_Player_can_Actions(env_state):
   arr_status = env_state[29:43]
   arr_Player_Action = np.where(arr_status == 1)[0]

   return arr_Player_Action

@njit()
def check_who_playing(arr_Player_Action,env_state):
   Player_action = env_state[43]
   check_coin_bet = env_state[64]
   if check_coin_bet == 0:
      Player_action = 0
      return Player_action
   if len(arr_Player_Action) == 1:
      Player_action = arr_Player_Action[0]
      return Player_action
   if Player_action == arr_Player_Action[-1]:
      Player_action = arr_Player_Action[0]
      return Player_action
   else:
      arr_Player_Action_choose = np.where(arr_Player_Action > Player_action)[0]
      idx_choose =  arr_Player_Action_choose[0]
      Player_action = arr_Player_Action_choose[idx_choose]
      return Player_action

@njit()
def Caculate_point_player(env_state,Player_action,P_player):
   point_player = 0
   if Player_action%2 == 0:
      deck_of_card = env_state[(44+P_player*22):(32+(P_player+1)*22)] 
   if Player_action%2 == 1:
      deck_of_card = env_state[(32+(P_player+1)*22):(42+(P_player+1)*22)]
   for idx_ in range(9):
      point_player += deck_of_card[idx_ + 1]*(idx_ + 2)
   if point_player >= 11:
      point_player += deck_of_card[0]*1
   if point_player < 11:
      point_player += deck_of_card[0]*11

   return point_player

@njit()
def check_cai_blackjack(env_state):
   player_status = env_state[31:43]       # trạng thái của 6 player còn lại
   if env_state[29] == 0 and player_status[11] != 0:
      return 1
   return 0

@njit()
def arr_blackjack_player(env_state):
   arr_blackjack = np.full(7,0)
   for idx in range(7):
      numbers_of_card_1 = np.sum(env_state[(44+idx*22):(32+(idx+1)*22)])
      numbers_of_card_2 = np.sum(env_state[(32+(idx+1)*22):(42+(idx+1)*22)])
      if numbers_of_card_1 == 2 and env_state[42+(idx+1)*22] == 21 and numbers_of_card_2 == 0:
         arr_blackjack[idx] = 1
   return arr_blackjack

@njit()
def stepEnv(action,env_state):
   arr_Player_Action = check_Player_can_Actions(env_state)

   if len(arr_Player_Action) != 0:
      Player_action = check_who_playing(arr_Player_Action,env_state)
      env_state[43] = Player_action
      P_player = int(env_state[43]//2)
      arr_card_on_board = env_state[:10]
      remainning = np.sum(env_state[:10])
      if remainning <= 8:
         env_state[:10] = [32,32,32,32,32,32,32,32,32,128]

      if action < 4:
         if action == 0:
            env_state[15+2*P_player] += 10 #tiền đặt
            env_state[10+P_player] -= 10   #tiền bị trừ đi
         if action == 1:
            env_state[15+2*P_player] += 20
            env_state[10+P_player] -= 20
         if action == 2:
            env_state[15+2*P_player] += 50
            env_state[10+P_player] -= 50
         if action == 3:
            env_state[15+2*P_player] += 100
            env_state[10+P_player] -= 100
         for i_ in range(2):
            idx_card_choose = weighted_random(arr_card_on_board)
            env_state[44+P_player*22 + idx_card_choose] += 1
            env_state[idx_card_choose+0] -= 1
         point_player = Caculate_point_player(env_state,Player_action,P_player)
         env_state[42+(P_player+1)*22] = point_player
      if action == 4:
         env_state[29+env_state[43]] = 0
      if action == 5:
         idx_card_choose = weighted_random(arr_card_on_board)
         if env_state[43]%2 == 0:
            env_state[44+P_player*22 + idx_card_choose] += 1
            env_state[idx_card_choose+0] -= 1
            point_player = Caculate_point_player(env_state,Player_action,P_player)
            env_state[42+(P_player+1)*22] = point_player
         if env_state[43]%2 == 1:
            env_state[32+(P_player+1)*22 + idx_card_choose] += 1
            env_state[idx_card_choose+0] -= 1
            point_player = Caculate_point_player(env_state,Player_action,P_player)
            env_state[43+(P_player+1)*22] = point_player
      if action == 6:
         env_state[15+2*P_player] *= 2
         env_state[10+P_player] -= int(env_state[15+2*P_player]/2)
         idx_card_choose = weighted_random(arr_card_on_board)
         env_state[44+P_player*22 + idx_card_choose] += 1
         env_state[idx_card_choose+0] -= 1
         point_player = Caculate_point_player(env_state,Player_action,P_player)
         env_state[42+(P_player+1)*22] = point_player
         env_state[29+env_state[43]] = 0
      if action == 7:
         env_state[16+2*P_player] = env_state[15+2*P_player]
         env_state[10+P_player]-= env_state[16+2*P_player]
         env_state[30+env_state[43]] = 1

         check = env_state[(44+P_player*22):(32+(P_player+1)*22)]
         idx_check = np.where(check == 2)[0]
         env_state[44+P_player*22 + idx_card_choose] = 1
         env_state[32+(P_player+1)*22 + idx_card_choose] = 1
         for i_ in range(2):
            idx_card_choose = weighted_random(arr_card_on_board)
            if i_ == 0:
               env_state[44+P_player*22 + idx_card_choose] += 1
               env_state[idx_card_choose+0] -= 1
               point_player = Caculate_point_player(env_state,Player_action,P_player)
               env_state[42+(P_player+1)*22] = point_player
            if i_ == 1:
               env_state[32+(P_player+1)*22 + idx_card_choose] += 1
               env_state[idx_card_choose+0] -= 1
               point_player = Caculate_point_player(env_state,Player_action,P_player)
               env_state[43+(P_player+1)*22] = point_player
         env_state[43] += 1
      if action == 8:
         check_player_action = np.sum(env_state[31:43])
         check_a = 1
         if env_state[64] == 0:  #nhà cái rút bài
            point_check = 0
            for i_ in range(2):
               idx_card_choose = weighted_random(arr_card_on_board)
               if i_ == 0:
                  env_state[44+idx_card_choose] += 1
                  env_state[idx_card_choose+0] -= 1
                  if idx_card_choose == 0:
                     env_state[64] = 11
                  if idx_card_choose != 0:
                     env_state[64] = idx_card_choose + 1
               if i_ == 1:
                  env_state[206] = idx_card_choose + 1
                  env_state[idx_card_choose+0] -= 1
               if idx_card_choose == 0:
                  point_check += 11
               if idx_card_choose != 0:
                  point_check += idx_card_choose + 1
            if point_check == 21:
               env_state[44:54] = [1,0,0,0,0,0,0,0,0,1]
               env_state[64] = 21
               env_state[206] = 0
               env_state[29] = 0
            check_a = 0
         if env_state[64] != 0 and check_a == 1:
            if check_player_action != 0:
               env_state[43] += 0
            if check_player_action == 0:
               env_state[43+env_state[206]] += 1
               point_player = Caculate_point_player(env_state,Player_action,P_player)
               env_state[64] = point_player
            if env_state[64] > 16:
               env_state[29] = 0
            if 0 < env_state[64] <= 16:
               idx_card_choose = weighted_random(arr_card_on_board)
               env_state[44+idx_card_choose] += 1
               env_state[idx_card_choose+0] -= 1
               point_player = Caculate_point_player(env_state,Player_action,P_player)
               env_state[64] = point_player
   
   ###-----------------------------------------------------------reset_small_game---------------------------------------------------------###
   ##Check_cai_blackjack###
   cai_blackjack = check_cai_blackjack(env_state)
   if cai_blackjack == 1:
      env_state[29:43] = 0
   
   point_end = env_state[np.array([64,65,86,87,108,109,130,131,152,153,174,175,196,197])]
   for zes in range(14):
      if point_end[zes] >= 21:
         env_state[29+zes] = 0
   ##reset_ván_chơi__###
   arr_Player_Action = check_Player_can_Actions(env_state)
   if len(arr_Player_Action) == 0:
      arr_blackjack = arr_blackjack_player(env_state)
      arr_vitual = np.full(7,1)
      arr_no_blackjack = arr_vitual - arr_blackjack
      blackjack = np.where(arr_blackjack == 1)[0]
      no_blackjack = np.where(arr_no_blackjack == 1)[0]
      if len(blackjack) != 0:       #TH1: Có người có blackjack
         if blackjack[0] == 0 and len(blackjack) == 1:      # BOT có blackjack
            for idx_choice in range(6):
               env_state[10] += int(env_state[17+2*idx_choice]*1.5)
               env_state[11+idx_choice] -= int(0.5*env_state[17+2*idx_choice])
               env_state[199] += 1
         if blackjack[0] == 0 and len(blackjack) > 1:
            blackjack_diff_bot = blackjack[1:]
            env_state[10+blackjack_diff_bot] += env_state[15+2*blackjack_diff_bot]
            env_state[199+blackjack] += 1
            for id in no_blackjack:
               if env_state[10+id] >= int(0.5*env_state[15+2*id]):
                  env_state[10] += int(1.5*env_state[15+2*id])
                  env_state[10+id] -= int(0.5*env_state[15+2*id])
               else:
                  env_state[10] += env_state[15+2*id] + env_state[10+id]
                  env_state[15+2*id] = 0
         if blackjack[0] != 0:   #bot_not_blackjack
            for idx_ in blackjack:
               env_state[10] -= int(1.5*(env_state[15+2*idx_]))
               env_state[10+idx_] += int(2.5*env_state[15+2*idx_])
               env_state[199+idx_] += 1

            not_blackjack = no_blackjack[1:]
            if point_end[0] <= 21:                 # nhà cái ít hơn 22đ
               for index_ in not_blackjack:
                  if point_end[2*index_] >= 22:
                     env_state[10] += env_state[15+2*index_]
                     env_state[199] += 1
                  if point_end[2*index_+1] >= 22:
                     env_state[10] += env_state[16+2*index_]
                     env_state[199] += 1
                  if point_end[2*index_] <= 21:
                     if point_end[2*index_] == env_state[64]:
                        env_state[10+index_] += env_state[15+2*index_]
                     elif point_end[2*index_] > env_state[64]:
                        env_state[10+index_] += 2*env_state[15+2*index_]
                        env_state[10] -= env_state[15+2*index_]
                        env_state[199+index_] += 1
                     elif point_end[2*index_] < env_state[64]:
                        env_state[10] += env_state[15+2*index_]
                        env_state[199] += 1
                  if point_end[2*index_+1] <= 21:
                     if point_end[2*index_+1] == env_state[64]:
                        env_state[10+index_] += env_state[16+2*index_]
                     elif point_end[2*index_+1] > env_state[64]:
                        env_state[10+index_] += 2*env_state[16+2*index_]
                        env_state[10] -= env_state[16+2*index_]
                        env_state[199+index_] += 1
                     elif point_end[2*index_+1] < env_state[64]:
                        env_state[10] += env_state[16+2*index_]
                        if point_end[2*index_+1] > 0:
                           env_state[199] += 1
            if point_end[0] >= 22:                  # Nhà cái lớn hơn 22đ
               for xct in not_blackjack:
                  if point_end[2*xct] >= 22:
                     env_state[10] += env_state[15+2*xct]
                  if point_end[2*xct+1] >= 22:
                     env_state[10] += env_state[16+2*xct]
                  if point_end[2*xct] <= 21:
                     env_state[10+xct] += 2*env_state[15+2*xct]
                     env_state[10] -= env_state[15+2*xct]
                     env_state[199+xct] += 1
                  if point_end[2*xct+1] <= 21:
                     env_state[10+xct] += 2*env_state[16+2*xct]
                     env_state[10] -= env_state[16+2*xct]
                     if point_end[2*xct+1] > 0:
                        env_state[199+xct] += 1
      else:
         not_blackjack = no_blackjack[1:]
         if point_end[0] <= 21:                 # nhà cái ít hơn 22đ
            for index_ in not_blackjack:
               if point_end[2*index_] >= 22:
                  env_state[10] += env_state[15+2*index_]
                  env_state[199] += 1
               if point_end[2*index_+1] >= 22:
                  env_state[10] += env_state[16+2*index_]
                  env_state[199] += 1
               if point_end[2*index_] <= 21:
                  if point_end[2*index_] == env_state[64]:
                     env_state[10+index_] += env_state[15+2*index_]
                  elif point_end[2*index_] > env_state[64]:
                     env_state[10+index_] += 2*env_state[15+2*index_]
                     env_state[10] -= env_state[15+2*index_]
                     env_state[199+index_] += 1
                  elif point_end[2*index_] < env_state[64]:
                     env_state[10] += env_state[15+2*index_]
                     env_state[199] += 1
               if point_end[2*index_+1] <= 21:
                  if point_end[2*index_+1] == env_state[64]:
                     env_state[10+index_] += env_state[16+2*index_]
                  elif point_end[2*index_+1] > env_state[64]:
                     env_state[10+index_] += 2*env_state[16+2*index_]
                     env_state[10] -= env_state[16+2*index_]
                     env_state[199+index_] += 1
                  elif point_end[2*index_+1] < env_state[64]:
                     env_state[10] += env_state[16+2*index_]
                     if point_end[2*index_+1] > 0:
                        env_state[199] += 1
         if point_end[0] >= 22:                  # Nhà cái lớn hơn 22đ
            for xct in not_blackjack:
               if point_end[2*xct] >= 22:
                  env_state[10] += env_state[15+2*xct]
               if point_end[2*xct+1] >= 22:
                  env_state[10] += env_state[16+2*xct]
               if point_end[2*xct] <= 21:
                  env_state[10+xct] += 2*env_state[15+2*xct]
                  env_state[10] -= env_state[15+2*xct]
                  env_state[199+xct] += 1
               if point_end[2*xct+1] <= 21:
                  env_state[10+xct] += 2*env_state[16+2*xct]
                  env_state[10] -= env_state[16+2*xct]
                  if point_end[2*xct+1] > 0:
                     env_state[199+xct] += 1
      env_state[17:29] = 0
      env_state[43:198] = 0
      env_state[206] = 0    
      for a_s in range(7):
         env_state[29 + a_s*2] = 1
      env_state[198] += 1   
           
   return env_state

@njit()
def getAgentsize():
   return 7

@njit()
def checkEnded(env_state):
   pointArr = env_state[10:17]
   if env_state[198] == 50:
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
         number_of_win_smallgame_1 = number_of_win_smallgame
         maxWin_smallgame_Play = np.where(number_of_win_smallgame_1 == maxWin_smallgame)[0]
         return maxWin_smallgame_Play[0]
   else:
      check_coin_end = np.where(pointArr >= 10)[0]
      if len(check_coin_end) == 1:
         maxpoin = np.max(pointArr)
         winpoint = np.where(pointArr == maxpoin)[0]
         return winpoint[0]
      else:
         return -1

@njit()
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

@njit()
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

def main(listAgent, times, perData):
   if len(listAgent) != 7:
      print('Hệ thống cho đúng 7 người chơi:>>>>')

   numWin = np.full(8,0)
   pIdOrder = np.arange(7)
   for w_ in range(times):
      # np.random.shuffle(pIdOrder)
      winner, perData = run([listAgent[pIdOrder[0]], listAgent[pIdOrder[1]], listAgent[pIdOrder[2]], listAgent[pIdOrder[3]], listAgent[pIdOrder[4]], listAgent[pIdOrder[5]], listAgent[pIdOrder[6]]], perData)

      if winner == -1:
         numWin[7] += 1
      else:
         numWin[pIdOrder[winner]] += 1
   return numWin, perData

@njit()
def numbaRun(p0,p1,p2,p3,p4,p5,p6,perData,pIdOrder):
   env_state = initEnv()
   __AGENT_SIZE__ = getAgentsize()
   tempData = []

   for _ in range(__AGENT_SIZE__):
      dataOnePlayer = List()
      dataOnePlayer.append(np.array([[0.]]))
      tempData.append(dataOnePlayer)

   winner = -1
   Id_player = int(env_state[43]//2)
   while env_state[198] <= 50:
      pIdx = int(env_state[43]//2)
      if pIdOrder[pIdx] == 0:
         action, tempData[pIdx], perData = p0(getAgentState(env_state), tempData[pIdx], perData)
      elif pIdOrder[pIdx] == 1:
         action, tempData[pIdx], perData = p1(getAgentState(env_state), tempData[pIdx], perData)
      elif pIdOrder[pIdx] == 2:
         action, tempData[pIdx], perData = p2(getAgentState(env_state), tempData[pIdx], perData)
      elif pIdOrder[pIdx] == 3:
         action, tempData[pIdx], perData = p3(getAgentState(env_state), tempData[pIdx], perData)
      elif pIdOrder[pIdx] == 4:
         action, tempData[pIdx], perData = p4(getAgentState(env_state), tempData[pIdx], perData)
      elif pIdOrder[pIdx] == 5:
         action, tempData[pIdx], perData = p5(getAgentState(env_state), tempData[pIdx], perData)
      elif pIdOrder[pIdx] == 6:
         action, tempData[pIdx], perData = p6(getAgentState(env_state), tempData[pIdx], perData)

      stepEnv(action,env_state)
      winner = checkEnded(env_state)
      if winner != -1:
         break
   for pIdx in range(7):
      Id_player = pIdx
      if pIdOrder[pIdx] == 0:
         action, tempData[pIdx], perData = p0(getAgentState(env_state), tempData[pIdx], perData)
      elif pIdOrder[pIdx] == 1:
         action, tempData[pIdx], perData = p1(getAgentState(env_state), tempData[pIdx], perData)
      elif pIdOrder[pIdx] == 2:
         action, tempData[pIdx], perData = p2(getAgentState(env_state), tempData[pIdx], perData)
      elif pIdOrder[pIdx] == 3:
         action, tempData[pIdx], perData = p3(getAgentState(env_state), tempData[pIdx], perData)
      elif pIdOrder[pIdx] == 4:
         action, tempData[pIdx], perData = p4(getAgentState(env_state), tempData[pIdx], perData)
      elif pIdOrder[pIdx] == 5:
         action, tempData[pIdx], perData = p5(getAgentState(env_state), tempData[pIdx], perData)
      elif pIdOrder[pIdx] == 6:
         action, tempData[pIdx], perData = p6(getAgentState(env_state), tempData[pIdx], perData)

   return winner, perData

@njit()
def numbaMain(p0,p1,p2,p3,p4,p5,p6,times, perData, printMode=False, k=50):
   numWin = np.full(8,0)
   __AGENT_SIZE__ = getAgentsize()
   pIdOrder = np.arange(__AGENT_SIZE__)
   for _ in range(times):
      if printMode and _ != 0 and _ % k == 0:
         print(_, numWin)
      np.random.shuffle(pIdOrder)
      winner, perData = numbaRun(p0,p1,p2,p3,p4,p5,p6, perData, pIdOrder)

      if winner == -1:
         numWin[7] += 1
      else:
         numWin[pIdOrder[winner]] += 1
         
   if printMode:
      print(_+1, numWin)

   return numWin, perData

def random_player(P_state, tempData, perData):
   ValidActions = getValidActions(P_state)
   ValidActions = np.where(ValidActions == 1)[0]
   print(ValidActions)
   action = np.random.choice(ValidActions)
   print(action)
   return action, tempData, perData

win, _ = main([random_player]*getAgentsize(), 1, [0])
print(win)