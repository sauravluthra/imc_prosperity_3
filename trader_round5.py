# SAURAV LUTHRA
# IMC PROSPERITY 3 - 2025
# TEAM: CAYO ALHAMBRARITHM

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import pandas as pd
import numpy as np
import statistics as stat
import math

# FOR VISUALIZER
import json
from typing import Any
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."
logger = Logger()

# =========== TRADER CLASS ===========
class Trader:

    # =======================================================================
    # ========================== INITIALIZE TRADER ==========================
    # =======================================================================
    def __init__(self):
    
        self.result        = {}
        self.observations  = {}
        self.tradingstate  = {}
        self.conversions   = 0

        self.positions     = {
                        'RAINFOREST_RESIN':            0, 
                        'KELP':                        0, 
                        'SQUID_INK':                   0,
                        'CROISSANTS':                  0,
                        'JAMS':                        0,
                        'DJEMBES':                     0,
                        'PICNIC_BASKET1':              0,
                        'PICNIC_BASKET2':              0,
                        'VOLCANIC_ROCK':               0,
                        'VOLCANIC_ROCK_VOUCHER_9500':  0,
                        'VOLCANIC_ROCK_VOUCHER_9750':  0,
                        'VOLCANIC_ROCK_VOUCHER_10000': 0,
                        'VOLCANIC_ROCK_VOUCHER_10250': 0,
                        'VOLCANIC_ROCK_VOUCHER_10500': 0,
                        'MAGNIFICENT_MACARON'        : 0
                        }
        
        self.position_lims = {'RAINFOREST_RESIN':       50, 
                        'KELP':                         50, 
                        'SQUID_INK':                    50,
                        'CROISSANTS':                  250,
                        'JAMS':                        350,
                        'DJEMBES':                       60,
                        'PICNIC_BASKET1':               60,
                        'PICNIC_BASKET2':              100,
                        'VOLCANIC_ROCK':               400,
                        'VOLCANIC_ROCK_VOUCHER_9500':  200,
                        'VOLCANIC_ROCK_VOUCHER_9750':  200,
                        'VOLCANIC_ROCK_VOUCHER_10000': 200,
                        'VOLCANIC_ROCK_VOUCHER_10250': 200,
                        'VOLCANIC_ROCK_VOUCHER_10500': 200,
                        'MAGNIFICENT_MACARON'        : 75
                        }

    # =======================================================================
    # ========================== RAINFOREST_RESIN ===========================
    # =======================================================================
    def rainforest_resin(self, data):

        # ========== SETUP ==========
        product = 'RAINFOREST_RESIN'
        position = self.positions[product]
        orders = []

        # ========== LOGIC ==========
        fair = 10000
        spreads = [2,2]

        orders.append(Order(product, fair + spreads[0], -50 - position))
        orders.append(Order(product, fair - spreads[1],  50 - position))

        # ========== OUTPUT ==========
        self.result[product] = orders

        return 0

    # =======================================================================
    # ================================ KELP =================================
    # =======================================================================
    def kelp(self, data):

        # ========== SETUP ==========
        product = 'KELP'
        position = self.positions[product]
        od = self.tradingstate.order_depths[product]
        orders = []
        
        # ========== LOGIC ==========
        tgt_posn = 0

        mid_price = self.mid_price(od)
        dev = mid_price - self.sma(data, 10)

        pctls = [-1.35,-1.10,-0.95,-0.80,-0.70,-0.60,-0.55,-0.50,-0.45,-0.40,-0.35,-0.35,
                 -0.30,-0.25,-0.20,-0.20,-0.15,-0.15,-0.10,-0.10,-0.05,-0.05,0.00,0.00,
                 0.00,0.05,0.05,0.05,0.10,0.10,0.10,0.15,0.15,0.20,0.20,0.25,0.30,0.30,0.35,
                 0.40,0.45,0.50,0.55,0.65,0.75,0.85,0.95,1.05,1.22, 1.25]
        
        idx = 0
        for pdx, p in enumerate(pctls):
            if dev > p:
                idx = pdx
        
        tgt_posn = max(-50, min(50, (-idx + 25) * 2))

        #logger.print(f'KELP ===== {mid_price}, {dev}, {idx}, {tgt_posn}\n')

        orders.append(Order(product, round(mid_price), round(tgt_posn - position)))

        # ========== OUTPUT ==========
        self.result[product] = orders
        
        return (data + [mid_price])[-10:]
    
    # =======================================================================
    # =========================== PICNIC_PRODUCTS ============================
    # =======================================================================
    def picnic_products(self, data):

        picnic_products = ['PICNIC_BASKET1', 'PICNIC_BASKET2', 'CROISSANTS', 'JAMS', 'DJEMBES']
        od = {}

        # get picnic products' mid_prices
        mid_prices = {}
        for product in picnic_products:
            od[product]         = self.tradingstate.order_depths[product]
            mid_prices[product] = self.mid_price(od[product])

        # ================ pb1, pb2, cro, jam, dje
        tgt_posns       = [0,   0,   0,   0,   0  ]

        # read in past data for moving averages
        past_spreads1 = data[0]
        past_spreads2 = data[1]

        # compute both picnic baskets' spreads
        spread1 = sum([1 * mid_prices['PICNIC_BASKET1'],
                      -6 * mid_prices['CROISSANTS'],
                      -3 * mid_prices['JAMS'],
                      -1 * mid_prices['DJEMBES'],
                      -39.07
                    ])
        
        spread2 = sum([1 * mid_prices['PICNIC_BASKET2'],
                      -4 * mid_prices['CROISSANTS'],
                      -2 * mid_prices['JAMS'],
                      -36.57
                    ])
        
        # update past spreads for traderData output
        data[0] = (past_spreads1 + [spread1])[-40:]
        data[1] = (past_spreads2 + [spread2])[-40:]
        
        # compute moving averages of spreads
        window1, window2 = 34, 25
        ma1 = self.sma(past_spreads1, window1)
        ma2 = self.sma(past_spreads2, window2)

        # compute moving average diffs as signals
        signal1 = spread1 - ma1
        signal2 = spread2 - ma2

        # z-scores of signals => scaled to trade more heavily on larger deviations
        # values scaled to [-1, 1]
        z1 = self.scale_zscore_sigmoid((signal1 + 0.0491) / 10.3767)
        z2 = self.scale_zscore_sigmoid((signal2 - 0.0014) / 5.4004)

        #order_sizes = self.picnic_order_sizes(z1, z2)
        tgt_posns[0] = -self.position_lims[picnic_products[0]] * z1
        tgt_posns[1] = -self.position_lims[picnic_products[1]] * z2

       
        tgt_posns[2] = self.clamp(-self.position_lims[picnic_products[2]], (-6 * tgt_posns[0]) + (-4 * tgt_posns[1]), -self.position_lims[picnic_products[2]])
        tgt_posns[3] = self.clamp(-self.position_lims[picnic_products[3]], (-3 * tgt_posns[0]) + (-2 * tgt_posns[1]), -self.position_lims[picnic_products[3]])
        tgt_posns[4] = self.clamp(-self.position_lims[picnic_products[4]], (-1 * tgt_posns[0]) + (-0 * tgt_posns[1]), -self.position_lims[picnic_products[4]])
      
        for idx, product in enumerate(picnic_products):
            self.result[product]= [Order(product, int(mid_prices[product]), round(tgt_posns[idx] - self.positions[product]))]
        # ========== OUTPUT ==========
         
        return data

    # =======================================================================
    # ========================== VOLCANIC_PRODUCTS ==========================
    # =======================================================================
    def volcanic_products(self, data):

        # ========== SETUP ==========
        underlying        =  'VOLCANIC_ROCK'
        volcanic_vouchers = ['VOLCANIC_ROCK_VOUCHER_9500',
                             'VOLCANIC_ROCK_VOUCHER_9750',
                             'VOLCANIC_ROCK_VOUCHER_10000',
                             'VOLCANIC_ROCK_VOUCHER_10250',
                             'VOLCANIC_ROCK_VOUCHER_10500'
                            ]
        strikes = [9500, 9750, 10000, 10250, 10500]
        
        underlying_data = data[0]
        a_data          = data[1]
        b_data          = data[2]
        c_dataIN        = data[3]
        
        # T - time to expiry
        ticks_per_year = 365000000
        expiry_tick    = 8000000
        curr_day       = 5
        curr_tick      = (curr_day * 1000000) + self.tradingstate.timestamp
        T              = (expiry_tick - curr_tick)/ticks_per_year

        # r - interest rate
        r = 0.0

        # ========= [underlying, 9500, 9750, 10000, 10250, 10500]
        tgt_posns = [0, 0, 0, 0, 0, 0]
        
        # dicts/lists mid_prices, implied vols, delta, log-moneyness, 
        mid_prices = {}
        impl_vol   = [0,0,0,0,0]
        log_mt     = [0,0,0,0,0]
        deltas     = [0,0,0,0,0]
        vegas      = [0,0,0,0,0]

        # UNDERLYING
        mid_prices[underlying] = self.mid_price(self.tradingstate.order_depths[underlying])
        S = mid_prices[underlying]
        ATM_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - S))
        real_vol = 0.1

        # VOUCHERS
        for idx, product in enumerate(volcanic_vouchers):
            
            K = int(product.rsplit("_", 1)[-1])
            mid_prices[product] = self.mid_price(self.tradingstate.order_depths[product])
            # get 5 voucher log_mt, impl_vols, deltas, and vegas
            log_mt[idx]   = self.log_mt(S, K, T)
            #impl_vol[idx] = self.impl_vol_voucher(mid_prices[product], S, K, T, r)
            impl_vol[idx] = self.impl_vol_bisection(mid_prices[product], S, K, T, r)
            deltas[idx]   = self.delta(S, K, T, r, max(impl_vol[idx], 0.0001))
            vegas [idx]   = self.vega (S, K, T, r, max(impl_vol[idx], 0.0001))

        # COEFFICIENTS - fitted implied volatility smile
        coeffs, curve_ivs = self.fit_vol_curve(log_mt, impl_vol)
        a, b, c = coeffs
        
        # predict C - Future ATM Implied Vol
        ma_coeffs         = [-0.8581, -0.0321, -0.0195]
        prev_c            = c_dataIN[0]
        prev_pred_delta_c = c_dataIN[1]
        err_c_1           = c_dataIN[2]
        err_c_2           = c_dataIN[3]

        delta_ct          = c - prev_c
        
        err_c             = delta_ct - prev_pred_delta_c
        
        pred_delta_c      = ma_coeffs[0]*err_c + ma_coeffs[1]*err_c_1 + ma_coeffs[2]*err_c_2
        c_dataOUT            = [c, pred_delta_c, err_c, err_c_1]

        # long/short calls/vouchers around the ATM implied vol, if it is expected to rise/fall
        if pred_delta_c > 0.00:
            #weights = self.atm_vol_weights(strikes=strikes, spot=S)
            weights = self.get_synthetic_call_weights(strikes=strikes, synthetic_strike=S)
            tgt_posns = [0] + [x * 100 for x in weights]
        elif pred_delta_c < 0.00:
            #weights = self.atm_vol_weights(strikes=strikes, spot=S)
            weights = self.get_synthetic_call_weights(strikes=strikes, synthetic_strike=S)
            tgt_posns = [0] + [x * -100 for x in weights]
        else:
            tgt_posns = [0,0,0,0,0,0]
        
        # delta-hedge all target voucher positions
        for idx, product in enumerate(volcanic_vouchers):
            tgt_posns[0] -= deltas[idx] * tgt_posns[idx + 1]
        
        # put all orders into self.result for each target position
        for idx, product in enumerate([underlying] + volcanic_vouchers):
            self.result[product] = [Order(product, round(mid_prices[product]), round(tgt_posns[idx] - self.positions[product]))]
    
        logger.print(
            f"IV:        {[f'{x:.2f},' for x in impl_vol]},\n"
            f"CURVE_IV:  {[f'{x:.2f},' for x in curve_ivs]},\n"
            f"COEFFS:    {[f'{x:.2f},' for x in coeffs]},\n"
            f"C_DATAIN:  {[f'{x:.2f},' for x in c_dataIN]},\n"
            f"C_DATAOUT: {[f'{x:.2f},' for x in c_dataOUT]},\n"
            f"WEIGHTS:   {[f'{x:.2f},' for x in weights]},\n"
            f"PRD_DLT_C: {f'{pred_delta_c:.2f},'},\n"
            f"TGT_VOUCH: {[f'{x:.2f},' for x in tgt_posns[1:]]},\n"
            f"TOT_DELTA: {tgt_posns[0]} ========"
        )
        # ========== OUTPUT ==========
        
        data = [underlying_data, a_data, b_data, c_dataOUT]

        return data
    
    # =======================================================================
    # ================================= RUN ================================= 
    # =======================================================================
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        
        if state.traderData.strip():  
            # if not empty or just whitespace
            traderData = json.loads(state.traderData)
        else:
            # default traderData dict
            traderData = [[],                    # rainforest_resin
                          [0,0,0,0,0,0,0,0,0,0], # kelp
                          [0,0,0,0,0,0,0,0,0,0], # squid_ink
                          [[],[]],               # picnic_products
                          [[],[],[],[0,0,0,0]],  # volcano_products
                          []]                    # magnificent_macarons

        products = state.order_depths
        self.tradingstate = state

        for product, position in state.position.items():
            self.positions[str(product)] = position
        
        self.observations = state.observations

        # ======================
        # run trading strategies
        # strategies append their orders to self.orders
        # strategies append their traderDataOUT to self.traderDataOUT

        traderData[0] = self.rainforest_resin    (traderData[0])
        traderData[1] = self.kelp                (traderData[1])
        #traderData[2] = self.squid_ink           (traderData[2])
        traderData[3] = self.picnic_products     (traderData[3])
        traderData[4] = self.volcanic_products   (traderData[4])
        #traderData[5] = self.magnificent_macarons (traderData[5])

        # ======================
        # FOR VISUALIZER
        logger.flush(state, self.result, self.conversions, json.dumps(traderData))
        
        return self.result, self.conversions, json.dumps(traderData)

    # =======================================================================
    # ===================== IMPLIED VOLATILITY FUNCTIONS ==================== 
    # =======================================================================

    def fit_vol_curve(self, x, y):
        # fit a parabolic volatility smile to the points. return coeff and curve values

        # x - time scaled log moneyness, y - implied volatility
        x = np.array(x)
        y = np.array(y)
        
        # ignores any points that have impl_vol == 0
        # don't use 0 implied vol points to build the smile
        mask = (y != 0)
        x_masked = x[mask]
        y_masked = y[mask]

        # Fit: y = ax^2 + bx + c
        coeffs = np.polyfit(x_masked, y_masked, deg=2)
        a, b, c = coeffs

        # Evaluate fitted polynomial at input x values
        y_fit = a * x**2 + b * x + c

        return [[a, b, c], y_fit.tolist()]
    
    def impl_vol_voucher(self, C_mkt, S, K, T, r, tol=1e-7, max_iter=100):
        # IMPLIED VOLATILITY of Euro call option (Volcanic Rock Vouchers)
        intrinsic = max(S - K, 0)
        if C_mkt <= intrinsic:
            return 0.00

        try:
            # try NEWTON-RAPHSON - implied volatility
            sigma = 0.10  # initial guess
            for i in range(max_iter):

                price = self.bs_call_price(S, K, T, r, sigma)
                v = self.vega(S, K, T, r, sigma)

                if v == 0:
                    raise ValueError("Zero Vega")

                diff = price - C_mkt

                if abs(diff) < tol:
                    return sigma

                sigma -= diff / v

                if sigma <= 0:
                    raise ValueError("Negative sigma")

        except Exception:
            # Newton-Raphson failed — use bisection as fallback
            vol = self.impl_vol_bisection(C_mkt, S, K, T, r, tol, max_iter)

            return vol

    def impl_vol_bisection(self, C_mkt, S, K, T, r, tol=1e-7, max_iter=100):
        # BISECTION METHOD - implied volatility
        # search range 0.001% - 100% implied vol

        intrinsic = max(S - K, 0)
        if C_mkt <= intrinsic:
            return 0.00
        
        low = 1e-5
        high = 1e1

        for _ in range(max_iter):

            mid = (low + high) / 2.0
            price = self. bs_call_price(S, K, T, r, mid)
            if abs(price - C_mkt) < tol:
                return mid
            if price > C_mkt:
                high = mid
            else:
                low = mid
        return mid  
        
    def bs_call_price(self, S, K, T, r, sigma):
        # BLACK-SCHOLES PRICE of Euro call option
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))

        return S * N(d1) - K * math.exp(-r * T) * N(d2)
    
    def log_mt(self, S, K, T):
        # get the time-scaled log-moneyness of the vouchers (call options)
        return (np.log(S/K) / np.sqrt(T))

    def vega(self, S, K, T, r, sigma):
        # VEGA of Euro call option
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        n = lambda x: (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x**2)

        return S * math.sqrt(T) * n(d1)
    
    def delta(self, S, K, T, r, sigma):
        if sigma == 0 or sigma == None:
            return 0
        else:
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            return self.N(d1)

    def get_synthetic_call_weights(self, strikes, synthetic_strike):
        """
        Given a list of strikes in order and a desired synthetic strike price (ATM=S),
        return a list of weights for the adjacent strike call options to replicate
        a synthetic call option at the synthetic strike.

        Assumes that the synthetic strike lies between two strikes in the list.
        Returns a list of weights (0 to 1) corresponding to each strike.

        Raises ValueError if the synthetic_strike is outside the range of provided strikes.
        """
        if synthetic_strike < strikes[0] or synthetic_strike > strikes[-1]:
            return [0,0,0,0,0]
        
        weights = [0.0] * len(strikes)

        for i in range(len(strikes) - 1):
            k1, k2 = strikes[i], strikes[i + 1]
            if k1 <= synthetic_strike <= k2:
                # Linear interpolation weights
                w2 = (synthetic_strike - k1) / (k2 - k1)
                w1 = 1 - w2
                weights[i] = w1
                weights[i + 1] = w2
                break

        return weights

    def atm_vol_weights(self, strikes, spot):
        """
        Given a list of strikes and a spot price,
        returns a dict of weights (sum to 1) that 
        interpolate exposure centered at spot price.
        """
        # Sort strikes just in case
        strikes = sorted(strikes)

        # Handle edge cases: spot below min or above max
        if spot <= strikes[0]:
            return {k: 1.0 if k == strikes[0] else 0.0 for k in strikes}
        elif spot >= strikes[-1]:
            return {k: 1.0 if k == strikes[-1] else 0.0 for k in strikes}

        # Find the two strikes that bracket the spot
        for i in range(len(strikes) - 1):
            k_low = strikes[i]
            k_high = strikes[i + 1]
            if k_low <= spot <= k_high:
                # Linear interpolation weights
                w_high = (spot - k_low) / (k_high - k_low)
                w_low = 1.0 - w_high
                weights = {k: 0.0 for k in strikes}
                weights[k_low] = w_low
                weights[k_high] = w_high
                return weights

        # Should not reach here
        raise ValueError("Spot does not fall within any strike interval.")
    # =======================================================================
    # =========================== HELPER FUNCTIONS ========================== 
    # =======================================================================
    
    def norm_cdf(self, z):
        """Approximate standard normal CDF using erf."""
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    def gmm_cdf(self, x, weights, means, stds):
        """Compute GMM CDF at point x."""
        return sum(
            w * self.norm_cdf((x - mu) / sigma)
            for w, mu, sigma in zip(weights, means, stds)
        )

    def inverse_norm_cdf(self, p):
        """Approximate the inverse CDF (probit function) of the standard normal distribution."""
        # Clamp p to avoid log(0) or division by 0
        p = max(min(p, 1 - 1e-10), 1e-10)
        
        # Coefficients in rational approximations
        a = [ 2.50662823884, -18.61500062529,  41.39119773534, -25.44106049637]
        b = [-8.47351093090,  23.08336743743, -21.06224101826,   3.13082909833]
        c = [0.3374754822726147, 0.9761690190917186,
            0.1607979714918209, 0.0276438810333863,
            0.0038405729373609, 0.0003951896511919,
            0.0000321767881768, 0.0000002888167364,
            0.0000003960315187]

        # Approximation for central region
        if 0.08 < p < 0.92:
            q = p - 0.5
            r = q * q
            num = (((a[3]*r + a[2])*r + a[1])*r + a[0])
            den = ((((b[3]*r + b[2])*r + b[1])*r + b[0])*r + 1)
            return q * num / den

        # Approximation for tails
        q = math.sqrt(-2 * math.log(p if p < 0.5 else 1 - p))
        x = c[0]
        for i in range(1, len(c)):
            x += c[i] * q**i
        return -x if p < 0.5 else x

    def position_sizing_voucher(self, x, voucher_idx=0, max_position=1.0):
        """
        Compute % of max position to allocate.
        Returns a signed float: +ve for long, -ve for short.
        """
        # x ==> deviation between voucher implied volatility and curve iv
        # voucher_idx ==> 0:9500, 1:9750 ... 4:10000
        weights = [[0.1754,0.1839,0.0592,0.1338,0.1011,0.0514,0.0074,0.1878,0.0636,0.0364], 
                [0.1920,0.0920,0.0576,0.1110,0.1521,0.0218,0.2113,0.0779,0.0016,0.0828], 
                [0.1502,0.1321,0.0438,0.1610,0.1272,0.0382,0.0163,0.0859,0.2035,0.0419], 
                [0.2578,0.1199,0.0742,0.0493,0.2127,0.0298,0.0807,0.0094,0.0250,0.1413], 
                [0.1602,0.1616,0.1487,0.0136,0.0267,0.0493,0.0073,0.0154,0.3037,0.1137]]

        means   = [[0.0019,-0.0113,0.0120,-0.0007,-0.0095,-0.0139,-0.0194,0.0006,0.0033,-0.0018], 
                [-0.0046,0.0305,-0.0318,0.0037,0.0254,0.0386,-0.0001,0.0205,-0.0522,-0.0090],  
                [0.0034,-0.0114,0.0229,-0.0012,-0.0084,-0.0147,0.0244,-0.0045,0.0011,0.0056], 
                [0.0020,-0.0149,-0.0019,-0.0116,0.0004,-0.0171,0.0032,-0.0245,-0.0057,-0.0134], 
                [-0.0013,0.0089,-0.0006,0.0099,-0.0031,0.0088,0.0157,-0.0002,-0.0008,0.0089]]

        stds    = [[0.0017,0.0018,0.0017,0.0017,0.0018,0.0021,0.0026,0.0017,0.0016,0.0017], 
                [0.0028,0.0037,0.0023,0.0030,0.0031,0.0072,0.0028,0.0033,0.0047,0.0029], 
                [0.0025,0.0026,0.0028,0.0025,0.0026,0.0034,0.0057,0.0028,0.0025,0.0027], 
                [0.0021,0.0022,0.0025,0.0024,0.0022,0.0026,0.0024,0.0028,0.0024,0.0021], 
                [0.0023,0.0017,0.0020,0.0026,0.0028,0.0017,0.0019,0.0022,0.0021,0.0018]]

        cdf = self.gmm_cdf(x, weights[voucher_idx], means[voucher_idx], stds[voucher_idx])
        z = self.inverse_norm_cdf(cdf)

        # Optional: cap the z-score to prevent over-leverage
        z = max(min(z, 4), -4)

        # Normalize to [-1, 1] and scale by max position
        position = -z / 4 * max_position  # negative z → long
        return position

    def mid_price(self, od):
        # get mid_price from OrderDepth od
        sells = od.sell_orders.items()
        buys = od.buy_orders.items()

        if len(buys) > 0 and len(sells) > 0:
            return (list(sells)[0][0] + list(buys)[0][0])/2.0
        
        elif len(buys) > 0:
            return list(buys)[0][0]
        
        elif len(sells) > 0:
            return list(sells)[0][0]
        
        else:
            return None
        
    def N(self, x):
        # CDF of standard normal using math.erf
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    def sma(self, past_prices, window):
        # get a simple moving average

        if len(past_prices) == 0:
            return 0.0
        
        if len(past_prices) < window:
            return np.mean(past_prices)
        
        return np.mean(past_prices[-window:])
    
    def scale_zscore_sigmoid(self, z_score):
        # Apply a sigmoid transformation and scale to [-1, 1]
            
        return 2 / (1 + np.exp(-z_score)) - 1
    
    def scale_zscore_tanh(self, z_score):
        # Apply tanh transformation which naturally maps to [-1, 1]
            
        return np.tanh(z_score)

    def clamp(self, min_val, x, max_val):
        # clamp an int inside a range
        return max(min_val, min(x, max_val))