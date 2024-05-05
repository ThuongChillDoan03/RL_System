# Code_Bot_splendor_VIS_DCT

# Lấy thông tin các nguyên liệu đang có trên bàn
def getStockOnBoard(p_state):
  return p_state[:6]

# Lấy thông tin các nguyên liệu đang có của bạn
def getMyStock(p_state):
  return p_state[6:12]

# Lấy thông tin các nguyên liệu mặc định của bạn
def getStockConst(p_state):
  return p_state[12:17]

# Lấy thông tin tất cả các thẻ đang có trên bàn
def getCardOnBoard(p_state):
  return p_state[18:102].reshape(12,7)

  if (len(stocks_places_can_choose) == 1) and (ingt_places[stocks_places_can_choose[0]] == -1) and (broad_stock[stocks_places_can_choose[0]] >=1):
    no_am = np.where(broad_stock>0)[0]
    if 1<=len(no_am)<=3:
      ingt_must_choose = no_am
    if len(no_am) > 3:
      ingt_must_choose.append(stocks_places_can_choose[0])
      while len(ingt_must_choose) <= 3:
        ix = 0
        if ix != stocks_places_can_choose[0]:
          ingt_must_choose.append(no_am[ix])
        ix += 1
  elif (len(stocks_places_can_choose) == 1) and (ingt_places[stocks_places_can_choose[0]] >= -2) and (broad_stock[stocks_places_can_choose[0]] >=2):
    ingt_must_choose.append(stocks_places_can_choose[0])
  elif (len(stocks_places_can_choose) == 2) and (broad_stock[stocks_places_can_choose[0]] >=1) and (broad_stock[stocks_places_can_choose[1]] >=1):
    no_am = np.where(broad_stock>0)[0]
    if 1<=len(no_am)<=3:
      ingt_must_choose = no_am
    if len(no_am) > 3:
      ingt_must_choose.append(stocks_places_can_choose[0])
      ingt_must_choose.append(stocks_places_can_choose[1])
      while len(ingt_must_choose) <= 3:
        ix = 0
        if (ix != stocks_places_can_choose[0]) and (ix != stocks_places_can_choose[0]):
          ingt_must_choose.append(no_am[ix])
        ix += 1
  elif (len(stocks_places_can_choose) == 3) and (broad_stock[stocks_places_can_choose[0]] >=1) and (broad_stock[stocks_places_can_choose[1]] >=1) and (broad_stock[stocks_places_can_choose[2]] >=1):
    ingt_must_choose.append(stocks_places_can_choose[0])
    ingt_must_choose.append(stocks_places_can_choose[1])
    ingt_must_choose.append(stocks_places_can_choose[2])
  else:
    no_am = np.where(broad_stock>0)[0]
    if 1<=len(no_am)<=3:
      ingt_must_choose = no_am
    if len(no_am) > 3:
      ingt_must_choose.append(no_am[0])
      ingt_must_choose.append(no_am[1])
      ingt_must_choose.append(no_am[2])