CREATE SEQUENCE public.experiment_id_seq
    INCREMENT 1
    START 16
    MINVALUE 1
    MAXVALUE 9223372036854775807
    CACHE 1;

ALTER SEQUENCE public.experiment_id_seq
    OWNER TO postgres;