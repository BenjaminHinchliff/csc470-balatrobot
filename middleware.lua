
--local Hook = require "hook"
--local Bot = require "bot"

Middleware = { }


function random_key(tb)
    local keys = {}
    for k in pairs(tb) do table.insert(keys, k) end
    return keys[math.random(#keys)]
end

function random_element(tb)
    local keys = {}
    for k in pairs(tb) do table.insert(keys, k) end
    return tb[keys[math.random(#keys)]]
end

function Middleware.add_event_sequence_recursive(events)
    if events == nil or #events <= 0 then
        return true
    end
    local head = table.remove(events, 1)

    G.E_MANAGER:add_event(Event({
        trigger = 'after',
        delay = head.delay,
        blocking = false,
        func = function()
            head.func(head.args)
            Middleware.add_event_sequence_recursive(events)
            return true
        end
    }))
end

local function pushbutton(button)
    if button and button.config and button.config.button then
        G.E_MANAGER:add_event(Event({
            trigger = 'after',
            delay = Bot.SETTINGS.action_delay,
            blocking = false,
            func = function()
                G.FUNCS[button.config.button](button)
                return true
            end
        }))
    end
end


Middleware.is_opening_booster = false
Middleware.prev_gamestate = -1


Middleware.BUTTONS = {
    -- Main Menu Buttons
    --MAIN_MENU_PLAY = nil,

    -- Start Run Buttons
    --START_RUN_PLAY = nil,

    -- Blind Phase Buttons
    SMALLBLIND_SELECT = nil,
    BIGBLIND_SELECT = nil,
    BOSSBLIND_SELECT = nil,
    SMALLBLIND_SKIP = nil,
    BIGBLND_SKIP = nil,
    --BOSS_REROLL = nil,

    -- Play Phase Buttons
    --PLAY_HAND = nil,
    --DISCARD_HAND = nil,
    CASH_OUT = nil,

    -- Shop Phase Buttons
    NEXT_ROUND = nil,
    REROLL = nil,

    -- Pack Phase Buttons
    --SKIP_PACK = nil,

    -- Game Over Buttons
    --GAME_OVER_MAIN_MENU = nil,
}

Middleware.firewhenready = { }

local function firewhenready(condition, func)
    for i = 1, #Middleware.firewhenready, 1 do
        if Middleware.firewhenready[i] == nil then
            Middleware.firewhenready[i] = {
                ready = condition,
                fire = func
            }
            return nil
        end
    end

    Middleware.firewhenready[#Middleware.firewhenready + 1] = {
        ready = condition,
        fire = func
    }
end

local function c_update()
    for i = 1, #Middleware.firewhenready, 1 do
        if Middleware.firewhenready[i] and Middleware.firewhenready[i].ready() then
            Middleware.firewhenready[i].fire()
            Middleware.firewhenready[i] = nil
        end
    end
end

local function c_onmainmenu()

    local function click_run_play_button()
        G.FUNCS.start_run(nil, {stake = Bot.SETTINGS.stake, seed = Bot.SETTINGS.seed, challenge = Bot.SETTINGS.challenge})
        return true
    end

    local function click_main_play_button()
        local _play_button = G.MAIN_MENU_UI:get_UIE_by_ID('main_menu_play')

        G.FUNCS[_play_button.config.button]({
            config = { }
        })
        G.FUNCS.exit_overlay_menu()

        return true        
    end

    Middleware.add_event_sequence_recursive({
        { func = click_main_play_button, delay = 3.0 },
        { func = click_run_play_button, delay = 1.0 }
    })
end

local function c_can_play_hand()

    local function click_play_hand()
        if G.buttons == nil then return true end

        local _play_button = UIBox:get_UIE_by_ID('play_button', G.buttons.UIRoot)
        if _play_button ~= nil and _play_button.config.button ~= nil then
           G.FUNCS[_play_button.config.button](_play_button)
        end
    end

    local function click_discard_hand()
        if G.buttons == nil then return true end

        local _discard_button = UIBox:get_UIE_by_ID('discard_button', G.buttons.UIRoot)
        if _discard_button ~= nil and _discard_button.config.button ~= nil then
            G.FUNCS[_discard_button.config.button](_discard_button)
        end
    end

    local function decide()

        local _action, _cards_to_play = Bot.select_cards_from_hand()

        local _events = { }
        for i = 1, #_cards_to_play, 1 do
            _events[i] = {
                func = function()
                    G.hand.cards[i]:click()
                end,
                delay = 0.5
            }
        end

        -- Option 1: Play Hand
        if _action == Bot.CHOICES.PLAY_HAND then
            _events[#_events+1] = { func = click_play_hand, delay = 2.0 }
        end

        -- Option 2: Discard Hand
        if _action == Bot.CHOICES.DISCARD_HAND then
            _events[#_events+1] = { func = click_discard_hand, delay = 2.0 } 
        end

        Middleware.add_event_sequence_recursive(_events)

    end

    decide()

end

local function c_can_choose_booster_cards(skip_button)

    local function click_deck_card(card)
        card:click()
    end

    local function click_skip_booster()
        G.FUNCS[skip_button.config.button](skip_button)
    end

    local function decide()

        local _action, _card, _hand_cards = Bot.select_booster_action(G.pack_cards.cards, G.hand.cards)

        if _action == Bot.CHOICES.SKIP_BOOSTER_PACK then
            Middleware.add_event_sequence_recursive({
                { func = click_skip_booster, delay = 5.0 }
            })
        elseif _action == Bot.CHOICES.SELECT_BOOSTER_CARD then

            local _events = { }

            -- Click each card from your deck first (only occurs if _pack_card is consumeable)
            for i = 1, #_hand_cards, 1 do
                _events[i] = {
                    func = function()
                        click_deck_card(_hand_cards[i])
                    end,
                    delay = 5.0
                }
            end

            -- Then select the booster card to activate
            _events[#_events+1] = {
                func = function()
                    if not _card then return end
                    _card:click()
                end,
                delay = 2.0
            }

            _events[#_events+1] = {
                func = function()
                    if not _card then return end

                    local _use_button = _card.children.use_button
                    local _buy_and_use_button = _card.children.buy_and_use_button
                    local _buy_button = _card.children.buy_button
                    local _use_and_sell_button = _card.children.use_and_sell_button
                    if _use_button then
                        local _node_index = _card.ability.consumeable and 2 or 1
                        G.FUNCS[_use_button.definition.nodes[_node_index].config.button](_use_button.definition.nodes[_node_index])
                    elseif _buy_and_use_button then
                        G.FUNCS[_buy_and_use_button.definition.config.button](_buy_and_use_button.definition)
                    elseif _buy_button then
                        G.FUNCS[_buy_button.definition.config.button](_buy_button.definition)
                    elseif _use_and_sell_button then
                        G.FUNCS[_use_and_sell_button.definition.config.button](_use_and_sell_button.definition)
                    end
                end,
                delay = 2.0
            }

            -- Once the pack is done, set can_skip_booster back to false
            _events[#_events+1] = {
                func = function()
                    Middleware.is_opening_booster = false
                end,
                delay = 10.0
            }

            Middleware.add_event_sequence_recursive(_events)
        end

    end

    decide()

end


local function c_shop()

    local _done_shopping = false

    local _b_can_round_end_shop = true
    local _b_can_reroll_shop = Middleware.BUTTONS.REROLL and Middleware.BUTTONS.REROLL.config and Middleware.BUTTONS.REROLL.config.button

    local _cards_to_buy = { }
    for i = 1, #G.shop_jokers.cards, 1 do
        _cards_to_buy[i] = G.shop_jokers.cards[i].cost <= G.GAME.dollars and G.shop_jokers.cards[i] or nil
    end

    local _vouchers_to_buy = { }
    for i = 1, #G.shop_vouchers.cards, 1 do
        _vouchers_to_buy[i] = G.shop_vouchers.cards[i].cost <= G.GAME.dollars and G.shop_vouchers.cards[i] or nil
    end

    local _boosters_to_buy = { }
    for i = 1, #G.shop_booster.cards, 1 do
        _boosters_to_buy[i] = G.shop_booster.cards[i].cost <= G.GAME.dollars and G.shop_booster.cards[i] or nil
    end

    local _choices = { }
    _choices[Bot.CHOICES.NEXT_ROUND_END_SHOP] = _b_can_round_end_shop
    _choices[Bot.CHOICES.REROLL_SHOP] = _b_can_reroll_shop
    _choices[Bot.CHOICES.BUY_CARD] = #_cards_to_buy > 0 and _cards_to_buy or nil
    _choices[Bot.CHOICES.BUY_VOUCHER] = #_vouchers_to_buy > 0 and _vouchers_to_buy or nil
    _choices[Bot.CHOICES.BUY_BOOSTER] = #_boosters_to_buy > 0 and _boosters_to_buy or nil
    
    local _action, _card = Bot.select_shop_action(_choices)

    if _action == Bot.CHOICES.NEXT_ROUND_END_SHOP then
        pushbutton(Middleware.BUTTONS.NEXT_ROUND)
        _done_shopping = true
    elseif _action == Bot.CHOICES.REROLL_SHOP then
        pushbutton(Middleware.BUTTONS.REROLL)
    elseif _action == Bot.CHOICES.BUY_CARD or _action == Bot.CHOICES.BUY_VOUCHER or  _action == Bot.CHOICES.BUY_BOOSTER then
        _card:click()

        local _use_button = _card.children.use_button
        local _buy_button= _card.children.buy_button
        if _use_button then
            G.FUNCS[_use_button.definition.config.button](_use_button.definition)
        elseif _buy_button then
            G.FUNCS[_buy_button.definition.config.button](_buy_button.definition)
        end
    end

    if not _done_shopping then
        firewhenready(function()
            return G.shop ~= nil and G.STATE_COMPLETE and G.STATE == G.STATES.SHOP
        end, c_shop)
    end
end


local function c_can_rearrange_jokers()
    Bot.rearrange_jokers()
end

local function c_can_rearrange_hand()
    Bot.rearrange_hand()
end

local function c_start_play_hand()
    Middleware.add_event_sequence_recursive({
        { func = c_can_rearrange_jokers, delay = 2.0 },
        { func = c_can_rearrange_hand, delay = 2.0 },
        { func = c_can_play_hand, delay = 2.0 }
    })
end

local function c_select_blind()

    local _blind_on_deck = G.GAME.blind_on_deck

    if _blind_on_deck == 'Boss' then
        pushbutton(Middleware.BUTTONS.BOSSBLIND_SELECT)
        return
    end

    local _choice = Bot.skip_or_select_blind(_blind_on_deck)

    local _button = nil
    if _choice == Bot.CHOICES.SELECT_BLIND then
        if _blind_on_deck == 'Small' then
            _button = Middleware.BUTTONS.SMALLBLIND_SELECT
        elseif _blind_on_deck == 'Big' then
            _button = Middleware.BUTTONS.BIGBLIND_SELECT
        end
    elseif _choice == Bot.CHOICES.SKIP_BLIND_SELECT_VOUCHER then
        if _blind_on_deck == 'Small' then
            _button = Middleware.BUTTONS.SMALLBLIND_SKIP
        elseif _blind_on_deck == 'Big' then
            _button = Middleware.BUTTONS.BIGBLIND_SKIP
        end
    end

    pushbutton(_button)
end


local function set_blind_select_buttons()
    local _blind_on_deck = G.GAME.blind_on_deck

    local _blind_obj = G.blind_select_opts[string.lower(_blind_on_deck)]
    local _select_button = _blind_obj:get_UIE_by_ID('select_blind_button')

    -- TODO fix going to next blind when previous blind select opens a pack
    _select_button.config = Hook.addonwrite(_select_button.config, function(...)
        local _t, _k, _v = ...
        if _k == 'button' and _v ~= nil then
            Middleware.can_select_blind = true
        end
    end)

    if _select_button.config.button then
        Middleware.can_select_blind = true
    end

    if _blind_on_deck == 'Boss' then
        Middleware.BUTTONS.BOSSBLIND_SELECT = _select_button
        return
    end

    local _skip_button = _blind_obj:get_UIE_by_ID('tag_'.._blind_on_deck).children[2]

    if _blind_on_deck == 'Small' then
        Middleware.BUTTONS.SMALLBLIND_SELECT = _select_button
        Middleware.BUTTONS.SMALLBLIND_SKIP = _skip_button
    elseif _blind_on_deck == 'Big' then
        Middleware.BUTTONS.BIGBLIND_SELECT = _select_button
        Middleware.BUTTONS.BIGBLIND_SKIP = _skip_button
    end
end


local function c_initgamehooks()
    -- Hooks break SAVE_MANAGER.channel:push so disable saving. Who needs it when you are botting anyway...
    G.SAVE_MANAGER = {
        channel = {
            push = function() end
        }
    }

    -- Detect when hand has been drawn
    G.GAME.blind.drawn_to_hand = Hook.addcallback(G.GAME.blind.drawn_to_hand, function(...)
        --Middleware.add_event_sequence_recursive({
         --   { func = c_can_rearrange_jokers, delay = 2.0 },
         --   { func = c_can_rearrange_hand, delay = 2.0 },
         --   { func = c_can_play_hand, delay = 2.0 }
       -- })
       return nil
    end)

    -- Hook button snaps
    G.CONTROLLER.snap_to = Hook.addcallback(G.CONTROLLER.snap_to, function(...)
        local _self = ...

        if _self and _self.snap_cursor_to.node and _self.snap_cursor_to.node.config and _self.snap_cursor_to.node.config.button then
            sendDebugMessage("SNAPTO: ".._self.snap_cursor_to.node.config.button)

            local _button = _self.snap_cursor_to.node
            local _buttonfunc = _self.snap_cursor_to.node.config.button

            if _buttonfunc == 'select_blind' and G.STATE == G.STATES.BLIND_SELECT then
                set_blind_select_buttons()
                c_select_blind()
            elseif _buttonfunc == 'cash_out' then
                Middleware.BUTTONS.CASH_OUT = _button
                pushbutton(Middleware.BUTTONS.CASH_OUT)
            elseif _buttonfunc == 'toggle_shop' and G.shop ~= nil then
                Middleware.BUTTONS.NEXT_ROUND = _button

                firewhenready(function()
                    return G.shop ~= nil and G.STATE_COMPLETE and G.STATE == G.STATES.SHOP
                end, c_shop)
            end
        end
    end)

    -- Set reroll availability
    G.FUNCS.can_reroll = Hook.addcallback(G.FUNCS.can_reroll, function(...)
        local _e = ...
        Middleware.BUTTONS.REROLL = _e
    end)

    -- Booster pack opening
    G.FUNCS.can_skip_booster = Hook.addcallback(G.FUNCS.can_skip_booster, function(...)
        local _e = ...
        if _e.config.button == 'skip_booster' then
            if not Middleware.is_opening_booster then
                Middleware.is_opening_booster = true

                if G.STATE == G.STATES.SPECTRAL_PACK or G.STATE == G.STATES.TAROT_PACK then
                    -- Wait for hand cards to be drawn
                    Middleware.add_event_sequence_recursive({
                        { func = c_can_rearrange_hand, delay = 10.0 },
                        {
                            func = function()
                                c_can_choose_booster_cards(_e)
                            end,
                            delay = 5.0
                        }
                    })
                else
                    c_can_choose_booster_cards(_e)
                end
            end
        end
    end)
end


Middleware.STATEFUNCS = { }
Middleware.STATEFUNCS[G.STATES.MENU] = c_onmainmenu

local function w_gamestate(...)
    local _t, _k, _v = ...

    if _k == 'STATE' then
        if Middleware.STATEFUNCS[_v] then
            Middleware.STATEFUNCS[_v]()
        end

        Middleware.prev_gamestate = _v
    end
end




function Middleware.hookbalatro()

    -- Start game from main menu
    G.start_run = Hook.addcallback(G.start_run, c_initgamehooks)
    G = Hook.addonwrite(G, w_gamestate)
    G.update = Hook.addcallback(G.update, c_update)

end

return Middleware