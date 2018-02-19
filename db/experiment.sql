-- Table: public.experiment

-- DROP TABLE public.experiment;

CREATE TABLE public.experiment
(
    id bigint NOT NULL DEFAULT nextval('experiment_id_seq'::regclass),
    name text COLLATE pg_catalog."default" NOT NULL,
    description text COLLATE pg_catalog."default",
    CONSTRAINT experiment_pkey PRIMARY KEY (id)
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.experiment
    OWNER to postgres;