local LSTM = {}

-- Creates one timestep of one LSTM
function LSTM.lstm(opt)
    -- placeholders for input
    local x = nn.Identity()()
    -- previous cell values
    local prev_c = nn.Identity()()
    -- previous hidden values
    local prev_h = nn.Identity()()

    -- function that spits out newly allocated blocks for building our graph
    function new_input_sum()
        -- transforms input
        local i2h = nn.Linear(opt.rnn_size, opt.rnn_size)(x)
        -- transforms previous timestep's output
        local h2h = nn.Linear(opt.rnn_size, opt.rnn_size)(prev_h)
        -- componentwise addition layer
        return nn.CAddTable()({i2h, h2h})
    end

    -- Define the different gates of the LSTM
    local in_gate      = nn.Sigmoid()(new_input_sum())
    local forget_gate  = nn.Sigmoid()(new_input_sum())
    local out_gate     = nn.Sigmoid()(new_input_sum())
    local in_transform = nn.Tanh()(new_input_sum())

    -- next cell value is addition of two componentwise products
    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    -- construct module. First arg is input, second is output
    return nn.gModule({x, prev_c, prev_h}, {next_c, next_h})
end

return LSTM