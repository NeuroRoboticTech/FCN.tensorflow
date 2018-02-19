-- Table: public.average_accuracies

-- DROP TABLE public.average_accuracies;

CREATE TABLE public.average_accuracies
(
    id bigint NOT NULL DEFAULT nextval('average_accuracies_id_seq'::regclass),
    experiment_id bigint NOT NULL,
    epoch bigint NOT NULL,
    accuracy real NOT NULL,
    training boolean,
    CONSTRAINT average_accuracies_pkey PRIMARY KEY (id)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.average_accuracies
    OWNER to postgres;