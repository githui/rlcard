# -*- coding: utf-8 -*-
''' Implement findfriends Dealer class
'''
import functools

from rlcard.utils import init_54_deck
from rlcard.games.findfriends.utils import cards2str, findfriends_sort_card

class FindfriendsDealer:
    ''' Dealer will shuffle, deal cards, and determine players' roles
    '''
    def __init__(self, np_random):
        '''Give dealer the deck

        Notes:
            1. deck with 54 cards including black joker and red joker
        '''
        self.np_random = np_random
        self.deck = init_54_deck()*3
        self.deck.sort(key=functools.cmp_to_key(findfriends_sort_card))
        self.leader = None

    def shuffle(self):
        ''' Randomly shuffle the deck
        '''
        self.np_random.shuffle(self.deck)

    def deal_cards(self, players):
        ''' Deal cards to players

        Args:
            players (list): list of findfriendsPlayer objects
        '''
        hand_num = (len(self.deck) - 6) // len(players)
        for index, player in enumerate(players):
            current_hand = self.deck[index*hand_num:(index+1)*hand_num]
            current_hand.sort(key=functools.cmp_to_key(findfriends_sort_card))
            player.set_current_hand(current_hand)
            player.initial_hand = cards2str(player.current_hand)

            # print(f'player {index} has current hand with {len(current_hand)} cards')

            # for h in current_hand:
            #     print(f'current hand {h}')

    def determine_role(self, players):
        ''' Determine landlord and peasants according to players' hand

        Args:
            players (list): list of findfriendsPlayer objects

        Returns:
            int: landlord's player_id
        '''
        # deal cards
        self.shuffle()
        self.deal_cards(players)
        # players[0].role = 'landlord'
        self.leader = players[0]
        # players[1].role = 'peasant'
        # players[2].role = 'peasant'
        #players[0].role = 'peasant'
        #self.landlord = players[0]

        ## determine 'landlord'
        #max_score = get_landlord_score(
        #    cards2str(self.landlord.current_hand))
        #for player in players[1:]:
        #    player.role = 'peasant'
        #    score = get_landlord_score(
        #        cards2str(player.current_hand))
        #    if score > max_score:
        #        max_score = score
        #        self.landlord = player
        #self.landlord.role = 'landlord'

        # give the 'landlord' the  three cards
        # self.landlord.current_hand.extend(self.deck[-3:])
        # self.landlord.current_hand.sort(key=functools.cmp_to_key(findfriends_sort_card))
        # self.landlord.initial_hand = cards2str(self.landlord.current_hand)
        return self.leader.player_id
