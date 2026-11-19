# Um pedido como foi enviado ao Claude Haiku 5.5

## Instruções (fixas)

```
Você é o módulo de compreensão de um assistente virtual. O usuário fala português de Portugal.

Para cada pedido do usuário, identifique:
1. a intenção: a ação que o usuário quer. Escolha exatamente uma da lista de intenções.
2. as entidades: os trechos do pedido que são argumentos dessa ação. Cada entidade tem um tipo da lista de tipos de entidade.

Regras para as entidades:
- Copie o texto de cada entidade exatamente como está no pedido, palavra por palavra. Não corrija, não traduza, não complete.
- Uma entidade é um trecho contínuo do pedido. Não junte palavras que não estão lado a lado.
- Liste as entidades na ordem em que aparecem no pedido.
- Se o pedido não tiver nenhuma entidade, devolva a lista vazia.
- Nem toda palavra importante é uma entidade. Siga o critério dos exemplos resolvidos.

Use só as intenções e os tipos de entidade das listas abaixo.

Responda com um objeto JSON com os campos "intent" e "slots". Cada item de "slots" tem "type" e "text".

Intenções:
- alarm_query
- alarm_remove
- alarm_set
- audio_volume_down
- audio_volume_mute
- audio_volume_other
- audio_volume_up
- calendar_query
- calendar_remove
- calendar_set
- cooking_query
- cooking_recipe
- datetime_convert
- datetime_query
- email_addcontact
- email_query
- email_querycontact
- email_sendemail
- general_greet
- general_joke
- general_quirky
- iot_cleaning
- iot_coffee
- iot_hue_lightchange
- iot_hue_lightdim
- iot_hue_lightoff
- iot_hue_lighton
- iot_hue_lightup
- iot_wemo_off
- iot_wemo_on
- lists_createoradd
- lists_query
- lists_remove
- music_dislikeness
- music_likeness
- music_query
- music_settings
- news_query
- play_audiobook
- play_game
- play_music
- play_podcasts
- play_radio
- qa_currency
- qa_definition
- qa_factoid
- qa_maths
- qa_stock
- recommendation_events
- recommendation_locations
- recommendation_movies
- social_post
- social_query
- takeaway_order
- takeaway_query
- transport_query
- transport_taxi
- transport_ticket
- transport_traffic
- weather_query

Tipos de entidade:
- alarm_type
- app_name
- artist_name
- audiobook_author
- audiobook_name
- business_name
- business_type
- change_amount
- coffee_type
- color_type
- cooking_type
- currency_name
- date
- definition_word
- device_type
- drink_type
- email_address
- email_folder
- event_name
- food_type
- game_name
- game_type
- general_frequency
- house_place
- ingredient
- joke_type
- list_name
- meal_type
- media_type
- movie_name
- movie_type
- music_album
- music_descriptor
- music_genre
- news_topic
- order_type
- person
- personal_info
- place_name
- player_setting
- playlist_name
- podcast_descriptor
- podcast_name
- radio_name
- relation
- song_name
- sport_type
- time
- time_zone
- timeofday
- transport_agency
- transport_descriptor
- transport_name
- transport_type
- weather_descriptor
```

## Mensagem

```
Exemplos resolvidos, parecidos com o pedido:

Pedido: qual é o último email que recebi
Resposta: {"intent": "email_query", "slots": []}

Pedido: hey quero bilhetes de comboio para odivelas na quarta-feira
Resposta: {"intent": "transport_ticket", "slots": [{"type": "place_name", "text": "odivelas"}, {"type": "date", "text": "quarta-feira"}]}

Pedido: qual é a minha agenda hoje
Resposta: {"intent": "calendar_query", "slots": [{"type": "date", "text": "hoje"}]}

Pedido: tweet longo tempo de espera e empregados não profissionais
Resposta: {"intent": "social_post", "slots": [{"type": "media_type", "text": "tweet"}]}

Pedido: cancelar a reunião de quinta feira da próxima semana às duas da tarde
Resposta: {"intent": "calendar_remove", "slots": [{"type": "event_name", "text": "reunião"}, {"type": "date", "text": "quinta feira da próxima semana"}, {"type": "time", "text": "duas da tarde"}]}

Pedido: qual é o meu último email
Resposta: {"intent": "email_query", "slots": []}

Pedido: vou ter que me preocupar com o limpa neves cobrir o meu carro na entrada de casa na terça feira
Resposta: {"intent": "weather_query", "slots": [{"type": "weather_descriptor", "text": "limpa neves cobrir o meu carro"}, {"type": "date", "text": "terça feira"}]}

Pedido: quanto tempo é que eu tenho de ferver um ovo
Resposta: {"intent": "cooking_recipe", "slots": [{"type": "cooking_type", "text": "ferver"}, {"type": "food_type", "text": "ovo"}]}

Pedido: como está o tempo hoje
Resposta: {"intent": "weather_query", "slots": [{"type": "date", "text": "hoje"}]}

Pedido: recomenda um filme que é lançado na última sexta feira
Resposta: {"intent": "recommendation_movies", "slots": [{"type": "event_name", "text": "filme"}, {"type": "date", "text": "última sexta feira"}]}

Agora resolva este:

Pedido: qual é o tempo na sexta feira
```

## Esquema da resposta

```json
{"type": "object", "properties": {"intent": {"type": "string", "enum": ["alarm_query", "alarm_remove", "alarm_set", "audio_volume_down", "audio_volume_mute", "audio_volume_other", "audio_volume_up", "calendar_query", "calendar_remove", "calendar_set", "cooking_query", "cooking_recipe", "datetime_convert", "datetime_query", "email_addcontact", "email_query", "email_querycontact", "email_sendemail", "general_greet", "general_joke", "general_quirky", "iot_cleaning", "iot_coffee", "iot_hue_lightchange", "iot_hue_lightdim", "iot_hue_lightoff", "iot_hue_lighton", "iot_hue_lightup", "iot_wemo_off", "iot_wemo_on", "lists_createoradd", "lists_query", "lists_remove", "music_dislikeness", "music_likeness", "music_query", "music_settings", "news_query", "play_audiobook", "play_game", "play_music", "play_podcasts", "play_radio", "qa_currency", "qa_definition", "qa_factoid", "qa_maths", "qa_stock", "recommendation_events", "recommendation_locations", "recommendation_movies", "social_post", "social_query", "takeaway_order", "takeaway_query", "transport_query", "transport_taxi", "transport_ticket", "transport_traffic", "weather_query"]}, "slots": {"type": "array", "items": {"type": "object", "properties": {"type": {"type": "string", "enum": ["alarm_type", "app_name", "artist_name", "audiobook_author", "audiobook_name", "business_name", "business_type", "change_amount", "coffee_type", "color_type", "cooking_type", "currency_name", "date", "definition_word", "device_type", "drink_type", "email_address", "email_folder", "event_name", "food_type", "game_name", "game_type", "general_frequency", "house_place", "ingredient", "joke_type", "list_name", "meal_type", "media_type", "movie_name", "movie_type", "music_album", "music_descriptor", "music_genre", "news_topic", "order_type", "person", "personal_info", "place_name", "player_setting", "playlist_name", "podcast_descriptor", "podcast_name", "radio_name", "relation", "song_name", "sport_type", "time", "time_zone", "timeofday", "transport_agency", "transport_descriptor", "transport_name", "transport_type", "weather_descriptor"]}, "text": {"type": "string"}}, "required": ["type", "text"], "additionalProperties": false}}}, "required": ["intent", "slots"], "additionalProperties": false}
```
